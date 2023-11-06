import numpy as np
import pandas as pd
from pandas import DataFrame


def self_pairwise(names):
    """
    Get a pairwise list from the given list of names

    Parameters
    ----------
    names: list
        The list of given names
    Returns
    -------
    list
        A list of pair set
    """
    pair_list = []
    for component1 in names:
        for component2 in names:
            if component1 != component2:
                pair = {component1, component2}
                if pair not in pair_list:
                    pair_list.append(pair)
    return pair_list


def feature_vector(name, df):
    vector = np.append(
        df[df.serial == name].iloc[0, 3:].values,
        df[df.serial == name].fps.values[0]
    )
    return vector


def get_triple_df(data_df, donor=(0, 11), acceptor=(12, 19), by='ea'):
    """
    Generating triple components data frame and extracting features vector from
    give data table.

    Each triple components sample has 3*4 electronic features and 3*2048
    fingerprints.
    So, the shape of the triple components vector will be (6156,)

    Parameters
    ----------
    data_df: DataFrame
        Calculated table including fingerprints and electronic features
    donor: tuple
        The range of donor index in the data table
    acceptor: tuple
        The range of acceptor index in the data table
    by: ['ea', 'ip', 'd_re_energy', 'a_re_energy']
        Name or list of names to sort by
    Returns
    -------
    DataFrame
        Edited table including sequence and features
    """
    # Get donors and acceptors table
    donor_df = data_df.loc[donor[0]:donor[1], :]
    acceptor_df = data_df.loc[acceptor[0]:acceptor[1], :]
    # Pairwise DA
    donor_pair = self_pairwise(donor_df.serial.to_list())
    acceptor_pair = self_pairwise(acceptor_df.serial.to_list())

    def extract_features(single, double):
        """
        Sorting molecules by certain feature and extracting features

        Parameters
        ----------
        single: str
            1 component
        double: set
            2 components
        Returns
        -------
        list
        """
        sorted_list, el, fps = [], [], []
        mol_list = [single] + list(double)
        # Sorting the molecule by EA
        sorted_list = data_df[data_df.serial.isin(mol_list)].sort_values(
            by=by, ascending=True
        ).serial.to_list()
        # Extract electronic and fingerprints features
        for sorted_mol in sorted_list:
            el += data_df[data_df.serial == sorted_mol].iloc[0, 3:7].to_list()
            fps += data_df[data_df.serial == sorted_mol].iloc[0, 2].tolist()
        return el + fps, sorted_list

    idx = 0
    df = pd.DataFrame(columns=['donors', 'acceptors', 'sequence', 'features'])
    # 2 Donors with 1 acceptor
    for donors in donor_pair:
        for acceptor in acceptor_df.serial:
            feature, sequence = extract_features(single=acceptor, double=donors)
            df.loc[idx, :] = {'donors': donors, 'acceptors': acceptor,
                              'sequence': sequence, 'features': feature}
            idx += 1
    # 1 Donor with 2 acceptors
    for donor in donor_df.serial:
        for acceptors in acceptor_pair:
            feature, sequence = extract_features(single=donor, double=acceptors)
            df.loc[idx, :] = {'donors': donor, 'acceptors': acceptors,
                              'sequence': sequence, 'features': feature}
            idx += 1
    return df


def load_exp_data(df, boundaries=('8_2', '2_8'), method='max'):
    """

    Parameters
    ----------
    df: DataFrame
    boundaries: tuple
    method: str

    Returns
    -------
    DataFrame
    """
    df_data = df.loc[:, boundaries[0]: boundaries[1]]
    if method == 'max':
        df.loc[:, 'max_h2'] = df_data.max(axis=1)
    elif method == 'median':
        df.loc[:, 'median_h2'] = df_data.median(axis=1)
    else:
        raise ValueError('Given method is not a proper method.')
    return df


def class_y(y, thresholds=(150,)):
    y_class = np.zeros(y.shape)
    for n_class in range(len(thresholds)):
        y_class[y >= thresholds[n_class]] = n_class + 1
    return y_class


def extract_xy(exp_df, feature_df, thresholds=(0, 800)):
    x = None
    for idx in exp_df[(exp_df.max_h2 < thresholds[1]) & (exp_df.max_h2 > thresholds[0])].idx:
        feature = feature_df.loc[idx, 'features']
        if x is None:
            x = feature
        else:
            x = np.vstack((x, feature))
    y = exp_df[(exp_df.max_h2 < thresholds[1]) & (exp_df.max_h2 > thresholds[0])].max_h2.values
    return x, y
