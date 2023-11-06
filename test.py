import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import cross_validate
from skopt.plots import plot_convergence
from src import data_calculation, kernel, ml, utils


def test_opt():
    X = np.load('./data/test_files/X.npy', allow_pickle=True)
    y = np.load('./data/test_files/y.npy', allow_pickle=True)
    y_class = utils.class_y(y=y, thresholds=(150,))
    param = {'C': 1}
    model = svm.SVC(kernel='precomputed', C=1/param['C'])
    gp_min = ml.opt_model(
        n_call=70, model=model, X=X,y_class=y_class, n_splits=5
    )
    plot_convergence(gp_min)
    print('finished')


def test_pair_dis():
    paired_df = pd.read_pickle('./data/pair_df.pickle')
    dft_d_df = pd.read_excel('./data/dft_results.xlsx', sheet_name='donor')
    donor_df = data_calculation.add_fps(
        dft_df=dft_d_df,
        mol2_path=Path('./data/gaussian/mol2_out')
    )
    fe_0 = np.array(paired_df.loc[0, 'features'])
    fe_1 = np.array(paired_df.loc[1, 'features'])
    distance = data_calculation.pairwise_distance(feature_0=fe_0, feature_1=fe_1)
    print('finish')


def test_build_tbf():
    scaled_data = pd.read_pickle('../data/scaled_mol_data.pkl')
    paired_df = utils.get_triple_df(scaled_data)
    print('finished')


def test_cv():
    X = np.load('./data/test_files/X.npy', allow_pickle=True)
    y = np.load('./data/test_files/y.npy', allow_pickle=True)
    y_class = utils.class_y(y=y, thresholds=(150,))
    param = {'gamma1': 0.3996, 'gamma2': 2.6803,
             'gamma3': 1.0030, 'gamma4': 1.3699,
             'C': 4.1202e-06}
    model = svm.SVC(
        kernel=kernel.kernel(
            param['gamma1'], param['gamma2'], param['gamma3'], param['gamma4']
        ),
        C=1 / param['C']
    )
    model.fit(X, y_class)
    scores = cross_validate(model, X, y_class, n_jobs=5)


if __name__ == '__main__':
    test_cv()
