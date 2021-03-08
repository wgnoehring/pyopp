#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ovito.modifiers import SpatialCorrelationFunctionModifier
from .DisplacementPostprocessingPipeline import DisplacementPostprocessingPipeline

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"

class DisplacementAutocorrelation(DisplacementPostprocessingPipeline):
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

    def __iter__(self):
        for i in range(self.num_frames):
            yield self.extract(i)

    def extract(self, i):
        """Compute and extract autocorrelation function in frame or file `i`.

        Parameters
        ----------
        i: int
            file `i` in the file range, or frame `i` in the frame range

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
        data = self._load(i)
        C_real = data.tables['correlation-real-space']
        C_reci = data.tables['correlation-reciprocal-space']
        rdf = data.tables['correlation-real-space-rdf']
        mean1 = data.attributes['CorrelationFunction.mean1']
        mean2 = data.attributes['CorrelationFunction.mean2']
        variance1 = data.attributes['CorrelationFunction.variance1']
        variance2 = data.attributes['CorrelationFunction.variance2']
        if self.direct_summation:
            C_real_neighbor = data.attributes['correlation-neighbor']
            rdf_neighbor = data.attributes['correlation-neighbor-rdf']
        # We calculate the autocorrelation, so means and variance
        # of both properties should be the same
        assert(np.isclose(mean1, mean2))
        assert(np.isclose(variance1, variance2))
        assert(np.isclose(variance1, covariance))
        covariance = data.attributes['CorrelationFunction.covariance']
        if self.direct_summation:
            return C_real, C_reci, rdf, mean1, covariance, C_real_neighbor, rdf_neighbor
        else:
            return C_real, C_reci, rdf, mean1, covariance, None, None
