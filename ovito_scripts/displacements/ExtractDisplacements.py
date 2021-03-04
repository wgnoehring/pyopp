#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exctract displacements"""
import warnings
import numpy as np
from .DisplacementPostprocessingPipeline import DisplacementPostprocessingPipeline

__author__ = "Wolfram Georg Nöhring"
__email__ = "wolfram.noehring@imtek.uni-freiburg.de"


class ExtractDisplacements(DisplacementPostprocessingPipeline):
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

    def postprocess_single(self, i):
        """Compute and extract particle displacements in frame or file `i`.

        Parameters
        ----------
        i: int
            file `i` in the file range, or frame `i` in the frame range

        Returns
        -------
        displacements: numpy.ndarray
            `[N, 3]` array, where `N` is the number of selected particles

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
        if self.periodicity is not None:
            data.cell_.pbc = [bool(i) for i in self.periodicity]
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
            order = np.self.argsort(identifiers)
            return np.take(displacements, order, axis=0)
        else:
            return displacements
