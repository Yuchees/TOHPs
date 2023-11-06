#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Yu Che'

import numpy as np
import pandas as pd
from scipy.spatial import distance

from pandas import DataFrame
from numpy import ndarray
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def add_fps(dft_df, mol2_path, radius=3, nbits=2048,
            selected_column=('IP', 'EA', 'd_re_energy', 'a_re_energy')):
    """
    Add bit fingerprints and selected data column to a new data frame.

    Parameters
    ----------
    dft_df: DataFrame
        Given data frame
    mol2_path: Path
        Molecular structure folder path, file must be mol2 format
    radius: int
        Fingerprints radius
    nbits: int
        Length of bit fingerprints
    selected_column: tuple
        Name of selected data columns in given data frame

    Returns
    -------
    df: DataFrame
        Generated new data frame
    """
    column = ['serial', 'smiles', 'fps', 'ip', 'ea',
              'd_re_energy', 'a_re_energy']
    df = pd.DataFrame(columns=column)
    for idx in dft_df.index:
        serial = dft_df.loc[idx, 'serial']
        mol = Chem.MolFromMol2File(str(mol2_path / (serial + '.mol2')))
        mol_fps = np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nbits
            )
        )
        mol_smiles = Chem.MolToSmiles(mol)
        df.loc[idx, :] = ([serial, mol_smiles, None] +
                          dft_df.loc[idx, list(selected_column)].tolist())
        df.loc[idx, 'fps'] = mol_fps
    df[['ip', 'ea', 'd_re_energy', 'a_re_energy']] = \
        df[['ip', 'ea', 'd_re_energy', 'a_re_energy']].astype('float')
    return df


def pairwise_distance(feature_0, feature_1, bounds=(12, 2048)):
    """
    Calculate distance between given two samples

    Parameters
    ----------
    feature_0: ndarray
        paired feature1
    feature_1: ndarray
        paired feature2
    bounds: tuple
        The length of electronic and fingerprints descriptors
    Returns
    -------
    ndarray
        (4,) array, distance between two given features,
        containing electronic distance and 3 molecules fingerprints distance
    """
    assert feature_0.shape == feature_1.shape, \
        'Given two samples feature does not have same shape.'
    # Calculate fingerprints boundaries of 3 different molecules
    fps_boundary = [bounds[0]]
    for i in range(3):
        fps_boundary.append(fps_boundary[i] + bounds[1])
    # Calculate electronic distance
    el_dis = distance.euclidean(feature_0[:bounds[0]], feature_1[:bounds[0]])
    # Calculate fingerprints distance
    fps_dis = np.zeros((3,))
    for i in range(3):
        fps_0 = feature_0[fps_boundary[i]: fps_boundary[i + 1]]
        fps_1 = feature_1[fps_boundary[i]: fps_boundary[i + 1]]
        fps_dis[i] = distance.jaccard(fps_0, fps_1)
    return np.hstack((np.array(el_dis), fps_dis))
