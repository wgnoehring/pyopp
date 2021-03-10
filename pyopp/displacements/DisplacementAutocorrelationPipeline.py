#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import numpy as np
from ovito.data import SimulationCell
from ovito.modifiers import (
    SpatialCorrelationFunctionModifier, 
    SliceModifier, 
    AffineTransformationModifier
)
from .DisplacementPostprocessingPipeline import DisplacementPostprocessingPipeline

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

logger = logging.getLogger('pyopp.displacements')

class DisplacementAutocorrelationPipeline(DisplacementPostprocessingPipeline):
    def __init__(
        self,
        component,
        apply_window=True,
        direct_summation=False,
        grid_spacing=3.0,
        neighbor_bins=50,
        neighbor_cutoff=5.0,
        *args,
        **kwargs
    ):
        """Initialize ExtractDisplacements.

        The parameters `apply_window`, `direct_summation`,
        `grid_spacing`, `neighbor_bins`, and `neighbor_cutoff` have
        the same meaning as the parameters with the same name of
        `ovito.modifiers.SpatialCorrelationFunctionModifier`

        Parameters
        ----------
        component : {"x", "y", "z"}
            Displacement component
        apply_window : bool, default=True
            Apply Hann window to nonperiodic directions
        direct_summation : bool, default=False
            Use direct summation instead of FFT
        grid_spacing : float, default=3.0
            Approximate size of FFT grid cells
        neighbor_bins : int, default=50
            Number of bins for direct calculation
        neighbor_cutoff : float, default=5.0
            Cutoff for the direct calculation
        """
        super().__init__(*args, **kwargs)
        self.component = component
        self.apply_window = apply_window
        self.direct_summation = direct_summation
        self.grid_spacing = grid_spacing
        self.neighbor_bins = neighbor_bins
        self.cutoff = neighbor_cutoff
        property_name = f"Displacement.{component.upper()}"
        m = SpatialCorrelationFunctionModifier(
            property1=property_name, property2=property_name,
            apply_window=self.apply_window,
            direct_summation=self.direct_summation,
            grid_spacing=self.grid_spacing,
            neighbor_bins=self.neighbor_bins,
            neighbor_cutoff=self.cutoff,
        )
        self.pipeline.modifiers.append(m)

    def _extract(self, data):
        """Extract spatial autocorrelation function data

        Parameters
        ----------
        data : ovito.DataCollection

        Returns
        -------
        C_real : DataTable
            Real-space autocorrelation function <u(x)u(x+dx)>.
        C_reci : DataTable
            Fourier-transformed autocorrelation function.
        rdf : DataTable
            Radial distribution function.
        mean1  : float
        covariance : float
        C_real_neighbor : None or DataTable
            Short-range part of real-space autocorrelation function,
            calculated using direct summation (only if `direct_summation=True`)
        rdf_neighbor : None or DataTable
            Short-range part of radial distribution function, 
            calculated using direct summation (only if `direct_summation=True`)
        """
        logger.info(f"{data.particles.count} particles")
        C_real = data.tables['correlation-real-space']
        C_reci = data.tables['correlation-reciprocal-space']
        rdf = data.tables['correlation-real-space-rdf']
        mean1 = data.attributes['CorrelationFunction.mean1']
        mean2 = data.attributes['CorrelationFunction.mean2']
        variance1 = data.attributes['CorrelationFunction.variance1']
        variance2 = data.attributes['CorrelationFunction.variance2']
        covariance = data.attributes['CorrelationFunction.covariance']
        if self.direct_summation:
            C_real_neighbor = data.tables['correlation-neighbor']
            rdf_neighbor = data.tables['correlation-neighbor-rdf']
        # We calculate the autocorrelation, so means and variance
        # of both properties should be the same
        assert(np.isclose(mean1, mean2))
        assert(np.isclose(variance1, variance2))
        assert(np.isclose(variance1, covariance))
        if self.direct_summation:
            return C_real, C_reci, rdf, mean1, covariance, C_real_neighbor, rdf_neighbor
        else:
            return C_real, C_reci, rdf, mean1, covariance, None, None


class DisplacementAutocorrelationSubvolumePipeline(DisplacementAutocorrelationPipeline):
    def __init__(self, up, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal = tuple(float(x) for x in (up == "x", up == "y", up == "z"))
        self.in_plane_indices, = np.where(np.logical_not(self.normal))
        self.normal_index = np.where(self.normal)[0][0]
        data = self._load(0)
        self._check_cell_shape(data.cell)
        length_along_normal, mean_length_in_plane = self._determine_distance_width(data.cell)
        is_autocorr_mod = tuple(
            isinstance(m, SpatialCorrelationFunctionModifier) for m in self.pipeline.modifiers
        )
        self.modifier_index = int(np.where(is_autocorr_mod)[0][0])
        self.slice_modifier = SliceModifier(
            distance=0.5*length_along_normal, 
            slab_width=mean_length_in_plane,
            normal=self.normal
        )
        self.pipeline.modifiers.insert(
            self.modifier_index, self.slice_modifier
        )
        self.affine_transformation_modifier = AffineTransformationModifier(
            relative_mode=False,
            target_cell=data.cell[...],
            operate_on={'cell'}
        )
        self.pipeline.modifiers.insert(
            self.modifier_index+1, self.affine_transformation_modifier
        )

    def _check_cell_shape(self, cell):
        triu_is_zero = np.all(np.isclose(np.triu(cell[:, :3], k=1), 0.0))
        tril_is_zero = np.all(np.isclose(np.tril(cell[:, :3], k=-1), 0.0))
        if not (triu_is_zero and tril_is_zero):
            logger.warning("simulation cell not orthogonal")

    def _determine_distance_width(self, cell):
        length_along_normal = cell[self.normal_index, self.normal_index]
        mean_length_in_plane = 0.5 * (
            cell[self.in_plane_indices[0], self.in_plane_indices[0]] + 
            cell[self.in_plane_indices[1], self.in_plane_indices[1]] 
        )
        return length_along_normal, mean_length_in_plane

    def _update_modifiers(self, data):
        self._check_cell_shape(data.cell)
        length_along_normal, mean_length_in_plane = self._determine_distance_width(data.cell)
        self.slice_modifier.distance = 0.5 * length_along_normal 
        self.slice_modifier.slab_width = mean_length_in_plane 
        logger.info(f"slice distance: {self.slice_modifier.distance:.2f}")
        logger.info(f"slab width: {self.slice_modifier.slab_width:.2f}")
        target_cell = data.cell_
        target_cell[self.normal_index, self.normal_index] = mean_length_in_plane  
        target_cell[self.normal_index, -1] = (
            data.cell[self.normal_index, -1] + 0.5 * (length_along_normal - mean_length_in_plane) 
        ) 
        self.affine_transformation_modifier.target_cell = target_cell
        logger.info(
            "transforming simulation cell:\n" +
            f"| {data.cell[0, 0]:7.1f} {data.cell[0, 1]:7.1f} {data.cell[0, 2]:7.1f} {data.cell[0, 3]:7.1f} |     | {target_cell[0, 0]:7.1f} {target_cell[0, 1]:7.1f} {target_cell[0, 2]:7.1f} {target_cell[0, 3]:7.1f} |\n" +
            f"| {data.cell[1, 0]:7.1f} {data.cell[1, 1]:7.1f} {data.cell[1, 2]:7.1f} {data.cell[1, 3]:7.1f} | --> | {target_cell[1, 0]:7.1f} {target_cell[1, 1]:7.1f} {target_cell[1, 2]:7.1f} {target_cell[1, 3]:7.1f} |\n" +
            f"| {data.cell[2, 0]:7.1f} {data.cell[2, 1]:7.1f} {data.cell[2, 2]:7.1f} {data.cell[2, 3]:7.1f} |     | {target_cell[2, 0]:7.1f} {target_cell[2, 1]:7.1f} {target_cell[2, 2]:7.1f} {target_cell[2, 3]:7.1f} |" 
        )
        return True
