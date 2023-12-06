#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kernel for HER learning, inspired by Daniele Padula.
Rewrite in Python 3.

References
----------
. [1] https://onlinelibrary.wiley.com/doi/10.1002/aenm.201902463
"""
__author__ = 'Yu Che'

import numpy as np

from src.data_calculation import pairwise_distance


def kernel(gamma1, gamma2, gamma3, gamma4):
    """
    Pre-defined gaussian kernel

    Parameters
    ----------
    gamma1: float
        Mixed electronic distance parameter
    gamma2: float
        Mol1 fingerprints distance parameter
    gamma3: float
        Mol2 fingerprints distance parameter
    gamma4: float
        Mol3 fingerprints distance parameter
    Returns
    -------
    RBF_kernel:
        Kernel function
    """
    def RBF_kernel(pair0, pair1):
        # Split electronic and fingerprints data from the flatten array
        dis = pairwise_distance(feature_0=pair0, feature_1=pair1)

        k = np.exp(-(
                gamma1 * dis[0] ** 2 +
                gamma2 * dis[1] ** 2 +
                gamma3 * dis[2] ** 2 +
                gamma4 * dis[3] ** 2
        ))
        return k
    return RBF_kernel
