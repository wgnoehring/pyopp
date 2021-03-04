#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline for computing and postprocessing particle displacements"""
import sys
import warnings
import argparse
from abc import ABC, abstractmethod
import pathlib
from functools import wraps
import numpy as np
from ovito.io import import_file
from ovito.modifiers import (
    CalculateDisplacementsModifier,
    UnwrapTrajectoriesModifier,
    ComputePropertyModifier,
)
from ovito.pipeline import FileSource
from ..util import parse_frame_range, xyz_to_num

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


class DisplacementPostprocessingPipeline(ABC):

    @abstractmethod
    def __init__(self, arguments=sys.argv[1:]):
        parser = self._create_parser()
        self.args = parser.parse_args(arguments)
        self._setup_pipeline()
        self._make_selection()

    def postprocess_displacements_all_configurations(self):
        """Postprocess all frames or files."""
        for i in range(self.num_frames):
            postprocess_displacements_single(i)

    @abstractmethod
    def postprocess_displacements_single_configuration(self, i):
        """Postprocess particle displacements.

        The exact postprocessing (e.g. root mean square displacements)
        can be implemented here. The displacements in frame or file 
        `i` can be computed using `calculate_displacements`. If another
        modifier is necessary, e.g. `SpatialCorrelationModifier`, then
        `_setup_pipeline` may be modified accordingly in the subclass.

        Parameters
        ==========
        i: int
            file `i` in the file range, or frame `i` in the frame range

        """
        pass

    def calculate_displacements(self, i, sort=False):
        """Calculate particle displacements in frame or file `i`. 

        Parameters
        ==========
        i: int
            file `i` in the file range, or frame `i` in the frame range
        sort: bool
            sort displacement array according to particle identifiers

        Returns
        =======
        displacements: numpy.ndarray
            `[N, M]` array, where `N` is the number of selected particles
            and `M` is the number of components specified via the 
            command line argument `displacement_component`.

        """
        if self.args.trajectory_style == "single":
            print(f"loading frame {self.args.frames[i]} from file {self.args.file}")
            data = self.pipeline.compute(self.args.frames[i])
        else:
            print(f"loading frame 0 from file {self.args.files[i].name}")
            self.pipeline.source.load(self.args.files[i].as_posix())
            data = self.pipeline.compute(0)
            if self.pipeline.source.num_frames > 1:
                message = f"more than one frame in {self.args.files[i].name}, will use frame 0"
                warnings.warn(message)
        if self.args.periodicity is not None:
            data.cell_.pbc = [bool(i) for i in self.args.periodicity]
        # find particles in selection
        if self.args.particle_subset is not None:
            (selection,) = np.where(
                np.isin(data.particles["Particle Identifier"].array, selected_ids)
            )
        else:
            selection = np.s_[0 : self.num_particles]
        columns = xyz_to_num[self.args.displacement_component]
        displacements = data.particles["Displacement"].array[selection, columns]
        if sort:
            identifiers = data.particles["Particle Identifier"].array[selection]
            order = np.self.argsort(identifiers)
            return np.take(displacements, order, axis=0)
        else:
            return displacements

    def _make_selection(self):
        """Define the IDs of selected particles

        Read file with particle identifiers that has been specified
        using the command line argument `--particle_subset`. If no
        file has been specified, then all particles are considered
        selected.

        """
        data = self.pipeline.compute(0)
        self.num_particles = data.particles.count
        if self.args.particle_subset is not None:
            self.selected_ids = np.load(self.args.particle_subset)
            self.num_selected = selected_ids.size
        else:
            self.num_selected = self.num_particles
        print(f"{self.num_particles} particles, {self.num_selected} selected")

    def _setup_pipeline(self):
        """Setup the pipeline for computing displacements.

        Sets up the Ovito pipeline with a `CalculateDisplacementsModifier`
        according to the arguments passed on the command, which should
        have been parsed using the parser from `_create_parser` before
        this function is called. If `--image_flags` has been set, then
        a `ComputePropertyModifier` will be added as well.

        """
        if self.args.trajectory_style == "single":
            self.pipeline = import_file(self.args.file.as_posix())
            self.num_frames = len(self.args.frames)
        else:
            self.pipeline = import_file(self.args.files[-1].as_posix())
            self.num_frames = len(self.args.files)
        data = self.pipeline.compute(0)
        if self.args.periodicity is not None:
            data.cell_.pbc = [bool(i) for i in self.args.periodicity]
        if self.args.image_flags is not None:
            self.pipeline.modifiers.append(
                ComputePropertyModifier(
                    output_property="Periodic Image", expressions=self.args.image_flags
                )
            )
        if self.args.trajectory_style == "single" and self.args.unwrap_trajectories:
            self.pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        if not "Particle Identifier" in data.particles:
            raise KeyError("cannot compute displacements without particle identifiers")
        m = CalculateDisplacementsModifier()
        if self.args.do_not_use_minimum_image_convention:
            m.minimum_image_convention = False
        if self.args.affine_mapping == "off":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.Off
        elif self.args.affine_mapping == "current":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToCurrent
        elif self.args.affine_mapping == "reference":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToReference
        else:
            raise ValueError(f"unknown affine_mapping style {affine_mapping}")
        if self.args.trajectory_style == "multiple":
            m.reference = FileSource()
            print(
                "importing reference configuration: {:s}".format(
                    self.args.reference_file.name
                )
            )
            m.reference.load(self.args.reference_file.as_posix())
            m.reference.compute(self.args.reference_frame)
        m.reference_frame = self.args.reference_frame
        self.pipeline.modifiers.append(m)

    def _create_parser(self):
        """Parse the command line

        Available subparsers (`trajectory_style` argument)
        * `single`
        * `multiple`

        Arguments available in `single` style:
        * `file`
        * `reference_frame`
        * `frames`
        * `displacement_component`
        * `--affine_mapping`
        * `--particle_subset`
        * `--periodicity`
        * `--image_flags`
        * `--do_not_use_minimum_image_convention`
        * `--unwrap_trajectories`

        Arguments available in `multiple` style:
        * `reference_file`
        * `--reference_frame`
        * `files`
        * `displacement_component`
        * `--affine_mapping"`
        * `--particle_subset"`
        * `--periodicity`
        * `--image_flags`
        * `--do_not_use_minimum_image_convention`

        Returns
        =======
        parser: argparse.ArgumentParser
            with subparsers `single` and `multiple` for analyzing trajectories
            that are stored in single or multiple files.
        """
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(
            dest="trajectory_style",
            required=True,
            help=(
                "'single': single trajectory file containing multiple frames "
                + "'multiple': multiple files with one frame per file. "
                + "Use 'single -h' or multiple '-h' to see further parameters."
            ),
        )
        subset_argument = dict(
            dest="particle_subset",
            type=pathlib.Path,
            help=(
                "numpy npy file containing atom identifiers for which to perform calculation"
            ),
        )
        non_affine_argument = dict(
            dest="affine_mapping",
            choices=["reference", "current", "off"],
            default="off",
            help="Subtract affine displacements",
        )
        component_argument = dict(
            dest="displacement_component",
            choices=["x", "y", "z", "norm", "xy", "xz", "yz", "yx", "zx", "zy"],
            help=(
                "Displacement component for which to perform VBM analysis."
                + "'norm' means the Euclidean norm of the three components."
            ),
        )
        periodicity_argument = dict(
            dest="periodicity",
            type=int,
            nargs=3,
            choices=[0, 1],
            default=None,
            help=(
                "Override periodic boundary condition settings. "
                + "One int for each boundary: 1 if boundary is periodic, 0 otherwise"
            ),
        )
        image_flags_argument = dict(
            dest="image_flags",
            type=str,
            nargs=3,
            default=None,
            help=(
                "Files may contain image flags, but Ovito may not recognize them. "
                + "E.g. the image flags in Lammps files may be called ix, iy, and iz."
                + "Use this option to specify the column names of image flags in x, y, and z."
                + "Note that you must specify all flags. You can also specify a number. "
                + "In this case, the corresponding flag of ALL particles will be set to this number. "
                + "Example: 'ix iy iz', 'ix 0 0', 'ix 0 iz'"
            ),
        )
        minimum_image_convention_argument = dict(
            dest="do_not_use_minimum_image_convention",
            action="store_true",
            help="Do not use minimum image convention when computing displacements.",
        )
        # Unwrapping trajectories is not supported if the trajectory style is multiple,
        # because an external file source has to be used for the displacement modifier,
        # and this external source will not be unwrapped.
        unwrap_trajectories_argument = dict(
            dest="unwrap_trajectories",
            action="store_true",
            help="Unwrap trajectories before computing displacements",
        )

        single = subparsers.add_parser("single")
        single.add_argument(
            "file",
            type=pathlib.Path,
            help="File containing the atomic configurations",
        )
        single.add_argument(
            "reference_frame",
            type=int,
            help="Reference frame for displacement calculation",
        )
        single.add_argument(
            "frames",
            type=parse_frame_range,
            help=(
                "Range of frames to analyse, with possible exceptions. "
                + "The most general format is 'start:end:increment-e1,e2,...', "
                + "where 'start' and 'end' are the limits (inclusive) of the range of frames "
                + "and 'increment' is the frame increment. The comma-separated list after the "
                "the semicolon contains the frames e1, e2, ... which should be skipped. "
                + "For example, '0:10:2-2,4' means that frames 0, 6, 8, and 10 are analysed. "
                + "The list of skipped frames can be omitted. The range can be also specified as "
                + "'start:end:' or 'start'. In the latter case, only one frame is analysed."
            ),
        )
        single.add_argument(**component_argument)
        single.add_argument("--affine_mapping", **non_affine_argument)
        single.add_argument("--particle_subset", **subset_argument)
        single.add_argument("--periodicity", **periodicity_argument)
        single.add_argument("--image_flags", **image_flags_argument)
        single.add_argument(
            "--do_not_use_minimum_image_convention", **minimum_image_convention_argument
        )
        single.add_argument("--unwrap_trajectories", **unwrap_trajectories_argument)

        multiple = subparsers.add_parser("multiple")
        multiple.add_argument(
            "reference_file",
            type=pathlib.Path,
            help=(
                "File which contains the frame that serves as "
                + "reference in the displacement computation. "
                + "This file may contain multiple frames. "
                + "By default, frame 0 will be used, but another "
                + "frame can be selected using 'reference_frame'."
            ),
        )
        multiple.add_argument(
            "files",
            type=pathlib.Path,
            nargs="*",
            help=(
                "List of files for which to perform calculation. "
                + "Each file should only contain one frame. Frame 0 "
                + "will be used for analysis. Each frame must contain "
                + "the same atoms identifiers as the reference frame."
            ),
        )
        multiple.add_argument(
            "-r",
            "--reference_frame",
            type=int,
            default=0,
            help="Frame number in reference_file",
        )
        multiple.add_argument(**component_argument)
        multiple.add_argument("--affine_mapping", **non_affine_argument)
        multiple.add_argument("--particle_subset", **subset_argument)
        multiple.add_argument("--periodicity", **periodicity_argument)
        multiple.add_argument("--image_flags", **image_flags_argument)
        multiple.add_argument(
            "--do_not_use_minimum_image_convention", **minimum_image_convention_argument
        )
        return parser
