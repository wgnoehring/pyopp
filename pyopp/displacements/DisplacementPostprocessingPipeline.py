#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline for computing and postprocessing particle displacements"""
import warnings
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


class DisplacementPostprocessingPipeline(object):
    """Ovito pipeline that computes and postprocesses displacements."""

    def __init__(
        self,
        files,
        frames=None,
        reference_frame=0,
        image_flags=None,
        minimum_image_convention=True,
        affine_mapping="off",
        unwrap_trajectories=False,
    ):
        """Initialize DisplacementPostprocessingPipeline

        Parameters
        ----------
        files : list of pathlib.Path
            List of files to be processed. If there is only one file, it will be
            interprested as a file containing multiple frames, and the parameter
            `reference_frame` defines the reference for the displacement
            calculation. Otherwise, the reference frame will be read from the
            first file, and displacements will be calculated for the other
            files.
        frames : list, optional
            If the parameter `files` contains a single file, then the list
            `frames` specifies the frames in that file for which displacements
            will be computed. Otherwise, the parameter is ignored.
        reference_frame : int, default=0
            number of the reference frame
        image_flags : list, optional
            Files may contain image flags, but Ovito may not recognize them.
            E.g. the image flags in Lammps files may be called `ix`, `iy`,
            and `iz`. Use this parameter to specify the column names of image
            flags in x, y, and z. Note that you must specify all flags. You can
            also specify a number (int or str) "0" or "1". In this case, the
            corresponding flag of ALL particles will be set to this number.
            Example: `['ix', 'iy', 'iz']` or `['ix', '0', '0']`, `['ix', 0, 0]`.
        minimum_image_convention : bool, default=True
            Use minimum image convention when computing displacements in
            periodic configurations?
        affine_mapping : {'off', 'current', 'reference'}, default="off"
            Choice for affine mapping before displacement calculation, see
            https://www.ovito.org/docs/current/particles.modifiers.displacement_vectors.php
        unwrap_trajectories : bool
            Unwrap coordinates of atoms that have crossed periodic boundaries before 
            computing displacements
        """

        self.files = files
        self.frames = frames
        self.reference_file = (files[0],)
        self.reference_frame = reference_frame
        self.image_flags = image_flags
        self.minimum_image_convention = minimum_image_convention
        self.affine_mapping = affine_mapping
        self.unwrap_trajectories = unwrap_trajectories
        self._setup_pipeline()

    def _setup_pipeline(self):
        """Setup the pipeline for computing displacements.

        Sets up the Ovito pipeline with a `CalculateDisplacementsModifier`
        according to the arguments passed on the command, which should
        have been parsed using the parser from `_create_parser` before
        this function is called. If `--image_flags` has been set, then
        a `ComputePropertyModifier` will be added as well.

        """
        single_file = len(self.files) == 1
        if single_file:
            self.pipeline = import_file(self.files[0].as_posix())
            self.num_frames = len(self.frames)
        else:
            self.pipeline = import_file(self.args.files[-1].as_posix())
            self.num_frames = len(self.files)
        data = self.pipeline.compute(0)
        if self.image_flags is not None:
            self.pipeline.modifiers.append(
                ComputePropertyModifier(
                    output_property="Periodic Image", 
                    expressions=tuple(str(f) for f in self.image_flags)
                )
            )
        if single_file and self.unwrap_trajectories:
            m = UnwrapTrajectoriesModifier()
            self.pipeline.modifiers.append(m)
        if not "Particle Identifier" in data.particles:
            raise KeyError("cannot compute displacements without particle identifiers")
        m = CalculateDisplacementsModifier()
        m.minimum_image_convention = self.minimum_image_convention
        if self.affine_mapping == "off":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.Off
        elif self.affine_mapping == "current":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToCurrent
        elif self.affine_mapping == "reference":
            m.affine_mapping = CalculateDisplacementsModifier.AffineMapping.ToReference
        else:
            raise ValueError(f"unknown affine_mapping style {affine_mapping}")
        if not single_file:
            m.reference = FileSource()
            print(
                "importing reference configuration: {:s}".format(
                    self.reference_file.name
                )
            )
            m.reference.load(self.reference_file.as_posix())
            m.reference.compute(self.reference_frame)
        m.reference_frame = self.reference_frame
        self.pipeline.modifiers.append(m)

    def __iter__(self):
        for i in range(self.num_frames):
            yield self.extract(i)

    def extract(self, i):
        data = self._load(i)
        self._update_modifiers(data)
        data = self._load(i)
        return self._extract(data)

    def _load(self, i):
        """Load frame and compute data

        Parameters
        ----------
        i: int
            file `i` in the file range, or frame `i` in the frame
            range. For example, if there is a single input file
            (`len(self.files)==1`) and `self.frames = [0, 1, 50, 99]`, then
            `_load(2)` will load and compute frame `50` in the input file.

        Returns
        -------
        data : ovito.DataCollection
        """
        single_file = len(self.files) == 1
        if single_file:
            print(f"loading frame {self.frames[i]} from file {self.files[0]}")
            data = self.pipeline.compute(self.frames[i])
        else:
            print(f"loading frame 0 from file {self.files[i].name}")
            self.pipeline.source.load(self.files[i].as_posix())
            data = self.pipeline.compute(0)
            if self.pipeline.source.num_frames > 1:
                message = (
                    f"more than one frame in {self.files[i].name}, will use frame 0"
                )
                warnings.warn(message)
        return data

    def _update_modifiers(self, data):
        """Update modifiers based on current data"""
        pass

    def _extract(self, data):
        return data
