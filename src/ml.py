import numpy as np
from numpy import ndarray

from sklearn import svm, metrics, neighbors
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.utils.parallel import Parallel

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from src.kernel import kernel

# model = svm.SVC(kernel='precomputed', C=1/param['C'], tol=1e-2)


class OptimizeModel(object):

    def __init__(self, model, X, y, space, n_splits, n_call):
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.n_call = n_call
        self.space = [Real(1e-8, 5, name='gamma1'),
                      Real(1e-8, 5, name='gamma2'),
                      Real(1e-8, 5, name='gamma3'),
                      Real(1e-8, 5, name='gamma4'),
                      space]

    def _svm_score(self, params):
        kf = KFold(self.n_splits)
        metric = kernel(params['gamma1'], params['gamma2'],
                        params['gamma3'], params['gamma4'])
        dis_matrix = metrics.pairwise.pairwise_kernels(self.X, self.X, metric)
        _score = []
        y_predict = np.array([])
        for train, test in kf.split(self.X):
            # Extract training, test matrix and y labels
            x_train = dis_matrix[train, :][:, train]
            x_test = dis_matrix[test, :][:, train]
            y_train = self.y[train]
            y_test = self.y[test]
            # train the model
            self.model.fit(x_train, y_train)
            if isinstance(self.model, svm.SVC):
                score = -self.model.score(x_test, y_test)
                _score.append(score)
            elif isinstance(self.model, svm.SVR):
                y_predict = np.hstack((y_predict, self.model.predict(x_test)))
            else:
                raise TypeError('')
        if isinstance(self.model, svm.SVR):
            _score = metrics.mean_squared_error(self.y, y_predict)
        return np.mean(_score)

    def _krr_score(self):
        return -cross_val_score(self.model, self.X, self.y, cv=self.n_splits,
                                n_jobs=-1, scoring='neg_mean_squared_error')

    def opt_param(self):
        @use_named_args(self.space)
        def objective(**params):
            return self._svm_score(params)

        gp_min = gp_minimize(objective, self.space, acq_func='EI',
                             n_calls=self.n_call, random_state=0)
        print("Best score={:4f}".format(gp_min.fun))
        return gp_min


def opt_model(n_call, model, X, y_class, n_splits=5, n_jobs=5):
    
    def svm_score(param):
        kf = KFold(n_splits)
        metric = kernel(param['gamma1'], param['gamma2'],
                        param['gamma3'], param['gamma4'])
        dis_matrix = metrics.pairwise.pairwise_kernels(X, X, metric)
        score_ = []
        y_predict = np.array([0])
        for train, test in kf.split(X):
            # Extract training, test matrix and y labels
            x_train = dis_matrix[train, :][:, train]
            x_test = dis_matrix[test, :][:, train]
            y_train = y_class[train]
            y_test = y_class[test]
            # train the model
            model.fit(x_train, y_train)
            if isinstance(model, svm.SVC) or isinstance(model, neighbors.KNeighborsClassifier):
                score = model.score(x_test, y_test)
                score_.append(score)
            elif isinstance(model, svm.SVR) or isinstance(model, neighbors.KNeighborsRegressor):
                 y_predict = np.hstack((y_predict, model.predict(x_test)))
            else:
                raise TypeError('')
        if isinstance(model, svm.SVR) or isinstance(model, neighbors.KNeighborsRegressor):
            score_ = -metrics.mean_squared_error(y_class, y_predict[1:])
        return -np.mean(score_)

    space = [Real(1e-8, 5, name='gamma1'),
             Real(1e-8, 5, name='gamma2'),
             Real(1e-8, 5, name='gamma3'),
             Real(1e-8, 5, name='gamma4'),
             Real(1e-8, 2, 'log-uniform', name='C')]

    @use_named_args(space)
    def objective(**params):
        return svm_score(params)

    gp_min = gp_minimize(objective, space, n_calls=n_call, random_state=0)
    print("Best score={:4f}".format(gp_min.fun))
    return gp_min
