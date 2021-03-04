#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ovito.modifiers import SpatialCorrelationFunctionModifier
from .DisplacementPostprocessingPipeline import DisplacementPostprocessingPipeline

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


class DisplacementAutocorrelation(DisplacementPostprocessingPipeline):
    def __init__(self, component, *args, **kwargs):
        """Initialize ExtractDisplacements.

        Parameters
        ----------
        selected_ids : numpy.ndarray
            Identifiers of atoms whose displacements should be returned.
            Note that displacements will be computed for all atoms.
        sort_by_id : bool
            sort displacement array according to particle identifiers
        """
        super().__init__(*args, **kwargs)
        self.component = component
        property_name = "Displacement.{component.upper()}"
        # Todo: further parameters for SpaticalCorrelationFunctionModifier
        m = SpatialCorrelationFunctionModifier(
            property1=property_name, property2=property_name
        )
        self.pipeline.modifiers.append(m)

    def postprocess_single(self, i):
        """Compute and extract particle displacements in frame or file `i`.

        Parameters
        ----------
        i: int
            file `i` in the file range, or frame `i` in the frame range

        """
        pass
