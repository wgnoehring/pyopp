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
        displacements: numpy.ndarray
            `[N, 3]` array, where `N` is the number of selected particles

        """
        data = self._load(i)
        data.cell_.pbc=(False,False,False)
        C_real = data.tables['correlation-real-space']
        C_reci = data.tables['correlation-reciprocal-space']
        rdf = data.tables['correlation-real-space-rdf']
        return C_real, C_reci, rdf
