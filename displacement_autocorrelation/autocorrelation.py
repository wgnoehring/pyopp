#!/home/fr/fr_fr/fr_wn1007/Tools/ovito-3.0.0-dev200-x86_64/bin/ovitos 
# -*- coding: utf-8 -*-
"""Calculate autocorrelation of non-affine displacements"""
import sys
import argparse
import numpy as np
import ovito
from ovito import modifiers
from ovito.io import import_file
from ovito.data import SimulationCell
from ovito.pipeline import FileSource

def main():
    # Get current configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'current_config', type=str, help=('Current configuration')
    )
    #parser.add_argument(
    #    'frame_number', type=int, help=('simulation frame')
    #)
    parser.add_argument(
        'reference_config', type=str, help=('Reference configuration')
    )
    parser.add_argument(
        'fft_grid_spacing', type=float, 
        help=('approximate grid spacing for FFT'), default=3.0
    )
    args = parser.parse_args()
    frame_number = int(float(args.current_config.rstrip(".nc")[-3::]))
    print("importing current configuration: \n{:s}".format(args.current_config))
    pipeline = import_file(args.current_config)
    data = pipeline.compute()
    cell = data.expect(SimulationCell)
    with cell:
        cell.pbc = (True, True, True)
     
    # Calculate non-affine displacements
    mod = modifiers.CalculateDisplacementsModifier()
    mod.reference = FileSource()
    print("importing reference configuration: \n{:s}".format(args.reference_config))
    mod.reference.load(args.reference_config)
    print("Take frame 0 as reference for displacement calculation")
    #mod.reference_frame = 0
    mod.affine_mapping = ovito.pipeline.ReferenceConfigurationModifier.AffineMapping.ToCurrent
    pipeline.modifiers.append(mod)
    data = pipeline.compute()

    # Calculate autocorrelation
    print("Setting up autocorrelation calculation with FFT grid spacing {:.3f}".format(
        args.fft_grid_spacing)
    )
    properties = ["Displacement.X", "Displacement.Y", "Displacement.Z"]
    prop = "Displacement.X"
    corr = modifiers.CorrelationFunctionModifier(
        property1=prop,
        property2=prop,
        grid_spacing=args.fft_grid_spacing,
        apply_window=True,
        #direct_summation=True,
        #neighbor_cutoff=12.0,
        #neighbor_bins=1024,
    )
    pipeline.modifiers.append(corr)
    data = pipeline.compute()
    for i, prop in enumerate(properties):
        print("calculating autocorrelation of {:s}".format(prop))
        pipeline.modifiers[-1].property1 = prop
        pipeline.modifiers[-1].property2 = prop
        data = pipeline.compute()
        prop_tag = "non-affine_" + prop.replace(".", "_").lower() + "_grid_size_{:.2f}_".format(args.fft_grid_spacing) + ".{:03d}.".format(frame_number)
        print("mean1, mean2, covariance: {:.8f} {:.8f} {:.8f}".format(
            corr.mean1, corr.mean2, corr.covariance)
        )
        reci = np.array(corr.get_reciprocal_space_function())
        real = np.array(corr.get_real_space_function())
        rdf  = np.array(corr.get_rdf())
        np.save("reci_autocorrelation_{:s}.npy".format(prop_tag), reci)
        np.save("real_autocorrelation_{:s}.npy".format(prop_tag), real)
        np.save("rdf_{:s}.npy".format(prop_tag), rdf)

if __name__ == "__main__":
    main()
