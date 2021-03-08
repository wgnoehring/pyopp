#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate autocorrelation of non-affine displacements"""
import pathlib
import click
from textwrap import dedent
import numpy as np
from ovito.io import export_file
from pyopp.util import parse_frame_range
from pyopp.displacements  import DisplacementAutocorrelation

# TODO: argument documentation
# TODO: grid_spacing, neighbor_bins, neighbor_cutoff

@click.group()
@click.argument(
    'component',
    type=click.Choice(
        ['x', 'y', 'z'], 
        case_sensitive=False
    )
)
@click.option(
    "--affine_mapping", 
    type=click.Choice(
        ['off', 'reference', 'current'], 
        case_sensitive=False
    ), 
    help=dedent(
        """\
        Subtract affine displacements by mapping the current configuration
        to the simulation cell of the reference configuration (reference),
        or by mapping the reference configuration to the cell of the current
        configuration (current).
        """
    ),
    default="off",
    show_default=True
)
@click.option(
    "--image_flags", 
    nargs=3, type=str,
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
    help="Apply Hann window to non-periodic directions"
)
@click.option(
    "--direct_summation/--no-direct_summation", 
    default=False,
    show_default=True,
    help="Use direct summation to calculate the autocorrelation function"
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
    )
)
@click.pass_context
def cli(ctx, component, affine_mapping, image_flags, window, direct_summation, minimum_image_convention):
    ctx.obj["component"] = component
    ctx.obj["affine_mapping"] = affine_mapping
    ctx.obj["image_flags"] = image_flags
    ctx.obj["window"] = window
    ctx.obj["direct_summation"] = direct_summation
    ctx.obj["minimum_image_convention"] = minimum_image_convention


@cli.command() 
@click.argument(
    "file", type=click.Path(exists=True), 
)
@click.argument(
    "reference_frame", type=int, 
)
@click.argument("frames", type=int, nargs=-1)
@click.option(
    "--unwrap_trajectories/--no-unwrap_trajectories", default=False,
    help="Unwrap trajectories before computing displacements",
)
@click.pass_context
def single(ctx, file, reference_frame, frames, unwrap_trajectories):
    if len(ctx.obj['image_flags']) == 3:
        image_flags = ctx.obj['image_flags']
    else:
        image_flags = None
    pipeline = DisplacementAutocorrelation(
        # Autocorrelation params
        component=ctx.obj['component'],
        apply_window=ctx.obj['window'],
        direct_summation=ctx.obj['direct_summation'],
        # Standard pipeline params
        files=(pathlib.Path(file),),
        frames=frames,
        reference_frame=reference_frame,
        image_flags=image_flags,
        minimum_image_convention=ctx.obj['minimum_image_convention'],
        affine_mapping=ctx.obj['affine_mapping'],
        unwrap_trajectories=unwrap_trajectories,
    )
    postprocess(pipeline)

@cli.command() 
@click.argument("reference_file", type=click.Path(exists=True))
@click.argument("files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--reference_frame", type=int, default=0,
    help="Frame number of reference configuration in REFERENCE_FILE",
)
@click.pass_context
def multi(ctx, reference_file, files, reference_frame):
    file_list = [pathlib.Path(reference_file)] + [pathlib.Path(f) for f in files]
    pipeline = DisplacementAutocorrelation(
        # Autocorrelation params
        component=ctx.obj['component'],
        apply_window=ctx.obj['window'],
        direct_summation=ctx.obj['direct_summation'],
        # Standard pipeline params
        files=file_list,
        frames=None,
        reference_frame=reference_frame,
        image_flags=image_flags,
        minimum_image_convention=ctx.obj['minimum_image_convention'],
        affine_mapping=ctx.obj['affine_mapping'],
    )
    postprocess(pipeline)

def postprocess(pipeline):
    for C_real, C_reci, rdf in pipeline:
        print(C_real)
        export_file(C_real, 'real_correlation_function.txt', 'txt/table')
        export_file(C_reci, 'reci_correlation_function.txt', 'txt/table')
        export_file(C_reci, 'rdf.txt', 'txt/table')

#def main():
#    args = parse_arguments()
#
#    for C_real, C_reci, rdf in pipeline:
#        export_file(C_real, 'real_correlation_function.txt', 'txt/table')
#        export_file(C_reci, 'reci_correlation_function.txt', 'txt/table')
#        export_file(C_reci, 'rdf.txt', 'txt/table')
#        #print("calculating autocorrelation of {:s}".format(prop))
#        #prop_tag = "non-affine_" + prop.replace(".", "_").lower() + "_grid_size_{:.2f}_".format(args.fft_grid_spacing) + ".{:03d}.".format(frame_number)
#        #print("mean1, mean2, covariance: {:.8f} {:.8f} {:.8f}".format(
#        #    corr.mean1, corr.mean2, corr.covariance)
#        #)
#        #reci = np.array(corr.get_reciprocal_space_function())
#        #real = np.array(corr.get_real_space_function())
#        #rdf  = np.array(corr.get_rdf())
#        #np.save("reci_autocorrelation_{:s}.npy".format(prop_tag), reci)
#        #np.save("real_autocorrelation_{:s}.npy".format(prop_tag), real)
#        #np.save("rdf_{:s}.npy".format(prop_tag), rdf)

if __name__ == "__main__":
    cli(obj={})