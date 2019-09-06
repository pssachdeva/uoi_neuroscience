"""
Performs a cross-validated coupling fit on various neuroscience datasets,
using glmnet from R.

This script performs and stores coupling models on this dataset using a
desired fitting procedure.
"""
import argparse
import h5py
import glmnet_python
import numpy as np

from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from neuropacks import ECOG, NHP, PVC11
from pyuoi.linear_model import UoI_Poisson
from sklearn.model_selection import StratifiedKFold


def log_likelihood(y_true, y_pred):
    return np.sum(y_true * np.log(y_pred) - y_pred)


def deviance(y_true, y_pred):
    ll_est = log_likelihood(y_true, y_pred)
    y_true_nz = y_true[y_true != 0]
    ll_true = log_likelihood(y_true_nz, y_true_nz)
    return ll_true - ll_est


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    n_folds = args.n_folds
    standardize = args.standardize
    verbose = args.verbose

    # extract dataset
    if args.dataset == 'ECOG':
        # create data extraction object
        ecog = ECOG(data_path=args.data_path)

        # get response matrix
        Y = ecog.get_response_matrix(
            bounds=(40, 60),
            band=args.band,
            electrodes=None,
            transform=None
        )
        class_labels = ecog.get_design_matrix(form='id')

    elif args.dataset == 'NHP':
        # create data extraction object
        nhp = NHP(data_path=args.data_path)

        # get response matrix
        Y = nhp.get_response_matrix(
            bin_width=args.bin_width,
            region=args.region,
            transform=args.transform
        )
        class_labels = None

    elif args.dataset == 'PVC11':
        # create data extraction object
        pvc = PVC11(data_path=args.data_path)

        # get response matrix
        Y = pvc.get_response_matrix(
            transform=args.transform
        )
        class_labels = pvc.get_design_matrix(form='label')

    else:
        raise ValueError('Dataset not available.')

    # clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) != 0).ravel()]
    n_targets = Y.shape[1]
    targets = np.arange(n_targets)

    # create fitter
    if args.fitter == 'UoI_Poisson':
        fitter = UoI_Poisson(
            n_lambdas=50,
            n_boots_sel=30,
            n_boots_est=30,
            selection_frac=0.8,
            estimation_frac=0.8,
            stability_selection=0.9,
            estimation_score='log',
            solver='lbfgs',
            standardize=True,
            fit_intercept=True,
            max_iter=10000,
            warm_start=False)

    # create folds
    skfolds = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=random_state)
    train_folds = {}
    test_folds = {}

    # create results dict
    intercepts = np.zeros((n_folds, n_targets))
    coupling_coefs = np.zeros((n_folds, n_targets, n_targets - 1))
    lls = np.zeros((n_folds, n_targets))
    deviances = np.zeros((n_folds, n_targets))

    # outer loop: create and iterate over cross-validation folds
    for fold_idx, (train_idx, test_idx) in enumerate(
        skfolds.split(y=class_labels, X=class_labels)
    ):
        if verbose:
            print('Fold %s' % fold_idx, flush=True)

        train_folds['fold_%s' % fold_idx] = train_idx
        test_folds['fold_%s' % fold_idx] = test_idx

        Y_train = Y[train_idx, :]
        Y_test = Y[test_idx, :]

        for target_idx, target in enumerate(targets):
            if verbose:
                print('Target ', target)

            # training design and response matrices
            X_train = np.delete(Y_train, target, axis=1)
            X_test = np.delete(Y_test, target, axis=1)
            y_train = Y_train[:, target]
            y_test = Y_test[:, target]

            # perform fit
            if args.fitter == 'glmnet':
                fit = cvglmnet(x=X_train, y=y_train, family='poisson',
                               nfolds=n_folds, standardize=standardize)
                coefs = cvglmnetCoef(fit, s='lambda_min').ravel()
                intercept = coefs[0]
                coef = coefs[1:]
            else:
                fitter.fit(X_train, y_train)
                intercept = fitter.intercept_
                coef = fitter.coef_

            intercepts[fold_idx, target_idx] = intercept
            coupling_coefs[fold_idx, target_idx] = coef

            # test design and response matrices
            y_pred = np.exp(intercept + np.dot(X_test, coef))
            lls[fold_idx, target_idx] = log_likelihood(y_test, y_pred)
            deviances[fold_idx, target_idx] = log_likelihood(y_test, y_pred)

    results_file = h5py.File(args.results_path, 'a')
    group = results_file.create_group(args.results_group)
    group['Y'] = Y
    group['coupling_coefs'] = coupling_coefs
    group['intercepts'] = intercepts
    group['lls'] = lls
    group['deviances'] = deviances

    train_folds_group = group.create_group('train_folds')
    test_folds_group = group.create_group('test_folds')

    for fold_key, fold_val in train_folds.items():
        train_folds_group[fold_key] = fold_val
    for fold_key, fold_val in test_folds.items():
        test_folds_group[fold_key] = fold_val

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--dataset')
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--fitter')

    # All datasets
    parser.add_argument('--transform', default=None)

    # NHP arguments
    parser.add_argument('--region', default='M1')
    parser.add_argument('--bin_width', type=float, default=0.25)

    # ECOG arguments
    parser.add_argument('--band', default='HG')

    # fitter object arguments
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--random_state', type=int, default=-1)

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
