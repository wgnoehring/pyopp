#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate autocorrelation of non-affine displacements"""
import sys
from os import remove
from datetime import  datetime
from io import BytesIO 
import logging
import tarfile
import pathlib
import click
from textwrap import dedent
import numpy as np
from ovito.io import export_file
from ovito import version_string as ovito_version
from pyopp import __version__ as pyopp_version
from pyopp.util import parse_frame_range
from pyopp.displacements import DisplacementAutocorrelationPipeline, DisplacementAutocorrelationSubvolumePipeline

logger = logging.getLogger('pyopp.scripts.autocorrelation')

@click.group()
@click.argument("component", type=click.Choice(["x", "y", "z"], case_sensitive=False))
@click.option(
    "--append_to_archive",
    type=bool,
    help="Append to tar archives instead of overwriting them",
    default=False,
    show_default=True,
)
@click.option(
    "--grid_spacing",
    type=float,
    help="Approximate size of FFT grid cell. See Ovito documentation.",
    default="3.0",
    show_default=True,
)
@click.option(
    "--neighbor_bins",
    type=int,
    help="Number of bins for direct calculation of real-space corrlation function. See Ovito documentation.",
    default="50",
    show_default=True,
)
@click.option(
    "--neighbor_cutoff",
    type=float,
    help="Cutoff for the direct calculation of the real-space correlation function. See Ovito documentation.",
    default="5.0",
    show_default=True,
)
@click.option(
    "--affine_mapping",
    type=click.Choice(["off", "reference", "current"], case_sensitive=False),
    help=dedent(
        """\
        Subtract affine displacements by mapping the current configuration
        to the simulation cell of the reference configuration (reference),
        or by mapping the reference configuration to the cell of the current
        configuration (current).
        """
    ),
    default="off",
    show_default=True,
)
@click.option(
    "--image_flags",
    nargs=3,
    type=str,
    help=dedent(
        """\
        Files may contain image flags, but Ovito may not recognize them. 
        E.g. the image flags in Lammps files may be called ix, iy, and iz.
        Use this option to specify the column names of image flags in x, y, and z.
        Note that you must specify all flags. You can also specify a number. 
        In this case, the corresponding flag of ALL particles will be set to this number. 
        Example: 'ix iy iz', 'ix 0 0', 'ix 0 iz'
        """
    ),
)
@click.option(
    "--window/--no-window",
    default=True,
    show_default=True,
    help="Apply Hann window to non-periodic directions",
)
@click.option(
    "--direct_summation/--no-direct_summation",
    default=False,
    show_default=True,
    help="Use direct summation to calculate the autocorrelation function",
)
@click.option(
    "--minimum_image_convention/--no-minimum_image_convention",
    default=True,
    show_default=True,
    help=dedent(
        """\
        Use minimum image convention when computing displacements of periodic
        configurations.
        """
    ),
)
@click.option(
    "--subvolume",
    type=click.Choice(["x", "y", "z"], case_sensitive=False),
    help=dedent(
        """\
        Calculate autocorrelation in a (approximately) cubic subvolume. This
        option was added for the special case of bi-axial compression, where
        the system shrinks in two directions, and elongates in the third
        direction. The cubic subvolume is created by slicing the configuration
        and shrinking the simulation cell. The value of this option is the
        direction along which the system will be sliced. For example, if this
        option is set to "z", then the configuration will be sliced in the
        z-direction to cut out a cubic subvolume in the center. The width of
        the slice is equal to the mean value of the cell lengths along the x-
        and y-directions. The simulation cell is then shrunk to enclose the
        subvolume.
        """
    ),
)
@click.pass_context
def cli(
    ctx,
    append_to_archive,
    component,
    grid_spacing,
    neighbor_bins,
    neighbor_cutoff,
    affine_mapping,
    image_flags,
    window,
    direct_summation,
    minimum_image_convention,
    subvolume
):
    """Calculate autocorrelation of displacements.

    This script calculates the spatial autocorrelation of
    the x-, y- or z-component of atomic displacements.
    The coordinates from which the displacements are
    computed can be provided in two ways. Command `single`
    reads the coordinates from a single file, e.g. a
    netCDF file or a Lammps dump file. Command `multi`
    reads the coordinates from multiple files (e.g.
    multiple Lammps dump or data files).

    The output is written to text files stored in tar
    archives. Three archives will be created (overwriting
    archives by the same name if they exist):
    `real_autocorrelation_COMPONENT_displacements.tar.gz`,
    `reci_autocorrelation_COMPONENT_displacements.tar.gz`,
    and `rdf_COMPONENT_displacements.tar.gz`.
    Each tar archive contains a file HEADER.txt with
    the parameters and version numbers.

    """
    ctx.obj["component"] = component
    ctx.obj["append_to_archive"] = append_to_archive
    ctx.obj["affine_mapping"] = affine_mapping
    ctx.obj["image_flags"] = image_flags
    ctx.obj["window"] = window
    ctx.obj["direct_summation"] = direct_summation
    ctx.obj["minimum_image_convention"] = minimum_image_convention
    ctx.obj["grid_spacing"] = grid_spacing
    ctx.obj["neighbor_bins"] = neighbor_bins
    ctx.obj["neighbor_cutoff"] = neighbor_cutoff
    ctx.obj["subvolume"] = subvolume


@cli.command()
@click.argument(
    "file",
    type=click.Path(exists=True),
)
@click.argument(
    "reference_frame",
    type=int,
)
@click.argument("frames", type=parse_frame_range)
@click.option(
    "--unwrap_trajectories/--no-unwrap_trajectories",
    default=False,
    help="Unwrap trajectories before computing displacements",
)
@click.pass_context
def single(ctx, file, reference_frame, frames, unwrap_trajectories):
    """Read coordinates from a single file.

    Compute the displacements from the atomic positions  in
    FILE. FILE must have one of the formats supported by Ovito,
    e.g. a netCDF file, or a Lammps dump file. It should contain
    one or more simulation frames, where each frame must contain
    the same number of atoms, and consistent atom identifiers.

    The displacements will be computed for each frame in the range
    FRAMES using REFERENCE_FRAME as the reference configuration.

    FRAMES is a string with the format `start:end-excluded_frames`,
    where `start` and `end` are integers, and `excluded_frames` is a
    comma-separated list of integers, e.g. `0:10:2-2,4` gives `0,6,8,10`.
    """
    if len(ctx.obj["image_flags"]) == 3:
        image_flags = ctx.obj["image_flags"]
    else:
        image_flags = None
    if ctx.obj["subvolume"] is None:
        pipeline = DisplacementAutocorrelationPipeline(
            # Autocorrelation params
            component=ctx.obj["component"],
            grid_spacing=ctx.obj["grid_spacing"],
            neighbor_bins=ctx.obj["neighbor_bins"],
            neighbor_cutoff=ctx.obj["neighbor_cutoff"],
            apply_window=ctx.obj["window"],
            direct_summation=ctx.obj["direct_summation"],
            # Standard pipeline params
            files=(pathlib.Path(file),),
            frames=frames,
            reference_frame=reference_frame,
            image_flags=image_flags,
            minimum_image_convention=ctx.obj["minimum_image_convention"],
            affine_mapping=ctx.obj["affine_mapping"],
            unwrap_trajectories=unwrap_trajectories,
        )
    else:
        pipeline = DisplacementAutocorrelationSubvolumePipeline(
            # Autocorrelation params
            slice_direction=ctx.obj["subvolume"],
            component=ctx.obj["component"],
            grid_spacing=ctx.obj["grid_spacing"],
            neighbor_bins=ctx.obj["neighbor_bins"],
            neighbor_cutoff=ctx.obj["neighbor_cutoff"],
            apply_window=ctx.obj["window"],
            direct_summation=ctx.obj["direct_summation"],
            # Standard pipeline params
            files=(pathlib.Path(file),),
            frames=frames,
            reference_frame=reference_frame,
            image_flags=image_flags,
            minimum_image_convention=ctx.obj["minimum_image_convention"],
            affine_mapping=ctx.obj["affine_mapping"],
            unwrap_trajectories=unwrap_trajectories,
        )
    ctx.obj["file"] = file
    ctx.obj["reference_frame"] = reference_frame
    ctx.obj["frames"] = frames
    ctx.obj["unwrap_trajectories"] = unwrap_trajectories
    postprocess(pipeline, ctx)


@cli.command()
@click.argument("reference_file", type=click.Path(exists=True))
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--reference_frame",
    type=int,
    default=0,
    help="Frame number of reference configuration in REFERENCE_FILE",
)
@click.pass_context
def multi(ctx, reference_file, files, reference_frame):
    """Read coordinates from multiple files.

    Read atomic coordinats in FILES and compute displacements,
    using the atomic coordinates of simulation frame
    REFERENCE_FRAME in REFERENCE_FILE  as
    reference configuration.

    All FILES must have one of the formats supported by Ovito, e.g. a
    netCDF file, or a Lammps dump file. Every file should contain only
    one simulation frame. If a file contains more than one frame, the
    displacements are calculated for frame zero. REFERENCE_FILE must
    be an Ovito-supported format as well, but it may contain multiple
    frames, where the desired reference configuration can be selected
    using the option `--reference_frame`.
    """
    file_list = [pathlib.Path(reference_file)] + [pathlib.Path(f) for f in files]
    if ctx.obj["subvolume"] is None:
        pipeline = DisplacementAutocorrelationPipeline(
            # Autocorrelation params
            component=ctx.obj["component"],
            grid_spacing=ctx.obj["grid_spacing"],
            neighbor_bins=ctx.obj["neighbor_bins"],
            neighbor_cutoff=ctx.obj["neighbor_cutoff"],
            apply_window=ctx.obj["window"],
            direct_summation=ctx.obj["direct_summation"],
            # Standard pipeline params
            files=file_list,
            frames=None,
            reference_frame=reference_frame,
            image_flags=image_flags,
            minimum_image_convention=ctx.obj["minimum_image_convention"],
            affine_mapping=ctx.obj["affine_mapping"],
        )
    else:
        pipeline = DisplacementAutocorrelationPipeline(
            # Autocorrelation params
            slice_direction=ctx.obj["subvolume"],
            component=ctx.obj["component"],
            grid_spacing=ctx.obj["grid_spacing"],
            neighbor_bins=ctx.obj["neighbor_bins"],
            neighbor_cutoff=ctx.obj["neighbor_cutoff"],
            apply_window=ctx.obj["window"],
            direct_summation=ctx.obj["direct_summation"],
            # Standard pipeline params
            files=file_list,
            frames=None,
            reference_frame=reference_frame,
            image_flags=image_flags,
            minimum_image_convention=ctx.obj["minimum_image_convention"],
            affine_mapping=ctx.obj["affine_mapping"],
        )
    postprocess(pipeline, ctx)


def postprocess(pipeline, ctx):
    context_string = ''.join(f'# {key} = {str(val)}{chr(10)}' for key, val in ctx.obj.items())
    header = dedent(
        f"""\
        # generated by pyopp version {pyopp_version}, with ovito version {ovito_version}
        # creation date: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')} 
        # command line: {sys.argv}
        # arguments and options:
        """ 
    ) + context_string
    archive_name_for_tag = dict()
    archive_name_for_tag["acf"] = f"real_autocorrelation_{pipeline.component}_displacements.tar"
    archive_name_for_tag["psd"] = f"reci_autocorrelation_{pipeline.component}_displacements.tar"
    archive_name_for_tag["rdf"] = f"rdf_{pipeline.component}_displacements.tar"
    if pipeline.direct_summation:
        archive_name_for_tag["acf_direct"] = f"real_autocorrelation_{pipeline.component}_displacements_direct.tar"
        archive_name_for_tag["rdf_direct"] = f"rdf_{pipeline.component}_displacements_direct.tar"
    extra_header_for_tag = dict()
    extra_header_for_tag["acf"] = '# "Distance r" C(r)\n'
    extra_header_for_tag["psd"] = '# "Wavevector q" C(q)\n'
    extra_header_for_tag["rdf"] = '# "Distance r" g(r)\n'
    if pipeline.direct_summation:
        extra_header_for_tag["acf_direct"] = '# "Distance r" "Neighbor C(r)"\n'
        extra_header_for_tag["rdf_direct"] = '# "Distance r" "Neighbor g(r)"\n'
    for tag, line in extra_header_for_tag.items():
        extra_header_for_tag[tag] = "# Meaning of columns in files:\n" + line
    archive_for_tag = dict()
    if ctx.obj["append_to_archive"]:
        mode = "a"
    else:
        mode = "w"
    for tag, name in archive_name_for_tag.items():
        archive_for_tag[tag] = tarfile.open(name, f"{mode}")
    # Write header to archives
    for tag, archive in archive_for_tag.items():
        tarinfo = tarfile.TarInfo('HEADER.txt')
        lines =  header + extra_header_for_tag[tag]
        tarinfo.size = len(lines)
        archive.addfile(tarinfo, BytesIO(lines.encode("utf8")))
        archive.close()
    statistics_file = open("mean_and_covariance.out", mode)
    statistics_file.write(header)
    statistics_file.write("# frame, mean, covariance\n")
    statistics_file.close()
    for frame, (acf, psd, rdf, mean, covariance, acf_direct, rdf_direct) in zip(pipeline.frames, pipeline):
        filename = f"frame_{frame:05d}.npy"
        logger.info(f"Mean of data: {mean:.12f}")
        logger.info(f"Covariance of data: {covariance:.12f}")
        statistics_file = open("mean_and_covariance.out", "a")
        statistics_file.write(f"{frame:5d} {mean:.12f} {covariance:.12f}\n")
        statistics_file.close()
        tables = (acf, psd, rdf, acf_direct, rdf_direct)
        tags = ("acf", "psd", "rdf", "acf_direct", "rdf_direct")
        for tag, table in zip(tags, tables):
            # Close archives to flush and immediately re-open in append mode.
            if table is None: continue
            destination = archive_name_for_tag[tag]
            logger.info(f"adding {filename} to {destination}")
            archive_for_tag[tag] = tarfile.open(destination, f"a")
            #export_file(table, filename, "txt/table")
            np.save(filename, table.xy())
            archive_for_tag[tag].add(filename, arcname=filename)
            remove(filename)
            archive_for_tag[tag].close()

if __name__ == "__main__":
    cli(obj={})
