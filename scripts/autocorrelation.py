#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate autocorrelation of non-affine displacements"""
import sys
from os import remove
from datetime import  datetime
from io import BytesIO 
import tarfile
import pathlib
import click
from textwrap import dedent
import numpy as np
from ovito.io import export_file
from ovito import version_string as ovito_version
from pyopp import __version__ as pyopp_version
from pyopp.util import parse_frame_range
from pyopp.displacements import DisplacementAutocorrelation

@click.group()
@click.argument("component", type=click.Choice(["x", "y", "z"], case_sensitive=False))
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
@click.pass_context
def cli(
    ctx,
    component,
    grid_spacing,
    neighbor_bins,
    neighbor_cutoff,
    affine_mapping,
    image_flags,
    window,
    direct_summation,
    minimum_image_convention,
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
    ctx.obj["affine_mapping"] = affine_mapping
    ctx.obj["image_flags"] = image_flags
    ctx.obj["window"] = window
    ctx.obj["direct_summation"] = direct_summation
    ctx.obj["minimum_image_convention"] = minimum_image_convention
    ctx.obj["grid_spacing"] = grid_spacing
    ctx.obj["neighbor_bins"] = neighbor_bins
    ctx.obj["neighbor_cutoff"] = neighbor_cutoff


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
    pipeline = DisplacementAutocorrelation(
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
    pipeline = DisplacementAutocorrelation(
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
    compression = "gz"
    tar_C_real_name = f"real_autocorrelation.tar_{pipeline.component}_displacements.tar.gz"
    tar_C_reci_name = f"reci_autocorrelation.tar_{pipeline.component}_displacements.tar.gz"
    tar_rdf_name = f"rdf_{pipeline.component}_displacements.tar.gz"
    tar_C_real = tarfile.open(tar_C_real_name, f"w:{compression}")
    tar_C_reci = tarfile.open(tar_C_reci_name, f"w:{compression}")
    tar_rdf = tarfile.open(tar_rdf_name, f"w:{compression}")
    tarinfo = tarfile.TarInfo('HEADER.txt')
    tarinfo.size = len(header)
    tar_C_real.addfile(tarinfo, BytesIO(header.encode("utf8")))
    tar_C_reci.addfile(tarinfo, BytesIO(header.encode("utf8")))
    tar_rdf.addfile(tarinfo,  BytesIO(header.encode("utf8")))
    for frame, (C_real, C_reci, rdf, mean1, mean2, covariance) in zip(pipeline. frames, pipeline):
        filename = f"frame_{frame:05d}.out"
        export_file(C_real, filename, "txt/table")
        tar_C_real.add(filename, arcname=filename)
        remove(filename)
        export_file(C_reci, filename, "txt/table")
        tar_C_reci.add(filename, arcname=filename)
        remove(filename)
        export_file(rdf, filename, "txt/table")
        tar_rdf.add(filename, arcname=filename)
        remove(filename)
    tar_C_real.close()
    tar_C_reci.close()
    tar_rdf.close()

if __name__ == "__main__":
    cli(obj={})
