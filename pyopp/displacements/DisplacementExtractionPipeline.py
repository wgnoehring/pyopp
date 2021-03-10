#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exctract displacements"""
import numpy as np
from .DisplacementPostprocessingPipeline import DisplacementPostprocessingPipeline

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


class DisplacementExtractionPipeline(DisplacementPostprocessingPipeline):
    def __init__(self, selected_ids=None, sort_by_id=False, *args, **kwargs):
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
        self.selected_ids = selected_ids
        self.sort_by_id = sort_by_id

    def _extract(self, data):
        """Extract displacements of selected particles

        Parameters
        ----------
        data : ovito.DataCollection

        Returns
        -------
        displacements: numpy.ndarray
            `[N, 3]` array, where `N` is the number of selected particles
        """
        # find particles in selection
        if self.selected_ids is not None:
            (selection,) = np.where(
                np.isin(data.particles["Particle Identifier"].array, selected_ids)
            )
        else:
            selection = np.s_[0 : data.particles.count]
        displacements = data.particles["Displacement"].array[selection, :]
        if self.sort_by_id:
            identifiers = data.particles["Particle Identifier"].array[selection]
            order = np.argsort(identifiers)
            return np.asarray(np.take(displacements, order, axis=0))
        else:
            return np.asarray(displacements)
