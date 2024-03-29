"""
Performs a cross-validated logistic regression on consonant-vowel syllables
using human ECoG data from the Chang lab.
"""
import argparse
import h5py
import numpy as np

from neuropacks import BG
from pyuoi.linear_model import UoI_L1Logistic
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    # process command line arguments
    data_path = args.data_path
    results_path = args.results_path
    results_group = args.results_group

    window_left = args.window_left
    window_right = args.window_right
    n_folds = args.n_folds
    with_std = args.with_std
    verbose = args.verbose

    # initialize neuropack
    bg = BG(data_path)
    good_units = bg.good_units
    n_good_units = good_units.size

    # get trial indices
    left_trials = bg.get_successful_left_trials()
    right_trials = bg.get_successful_right_trials()
    valid_trials = np.sort(np.concatenate((left_trials, right_trials)))
    n_valid_trials = valid_trials.size

    # storage arrays
    baseline_scores = np.zeros(n_folds)
    baseline_intercepts = np.zeros(n_folds)
    baseline_coefs = np.zeros((n_folds, n_good_units))
    uoi_scores = np.zeros(n_folds)
    uoi_intercepts = np.zeros(n_folds)
    uoi_coefs = np.zeros((n_folds, n_good_units))

    ###################
    # Preprocess data #
    ###################

    # create design and response matrices
    X = np.zeros((n_valid_trials, n_good_units))
    y = np.zeros(n_valid_trials)

    for t_idx, trial_idx in enumerate(valid_trials):
        trial = bg.trials[trial_idx]
        t_center_out = trial.t_center_out

        for u_idx, unit_idx in enumerate(good_units):
            spike_times = trial.spike_times[unit_idx]
            spike_count = np.count_nonzero(
                (spike_times >= t_center_out - window_left) &
                (spike_times <= t_center_out + window_right)
            )
            X[t_idx, u_idx] = spike_count

        y[t_idx] = trial.is_successful_right()

    # zero mean and standardize if necessary
    scaler = StandardScaler(with_std=with_std)
    X_new = scaler.fit_transform(X)

    ###################
    # Perform fitting #
    ###################

    # stratify the folds
    skf = StratifiedKFold(n_splits=n_folds,
                          shuffle=True,
                          random_state=random_state)
    skf.get_n_splits(X_new, y)

    # iterate over folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_new, y)):
        if verbose:
            print('Fold %s' % fold)

        # run vanilla logistic regression
        logistic = LogisticRegressionCV(
            Cs=args.n_Cs,
            fit_intercept=True,
            cv=args.cv,
            penalty='l1',
            solver='saga',
            max_iter=10000)
        logistic.fit(X_new[train_idx, :], y[train_idx])
        baseline_coefs[fold] = logistic.coef_
        baseline_intercepts[fold] = logistic.intercept_
        baseline_scores[fold] = logistic.score(X_new[test_idx, :],
                                               y[test_idx])

        # run UoI logistic regression
        uoi = UoI_L1Logistic(
            n_C=args.n_Cs,
            standardize=False,
            n_boots_sel=args.n_boots_sel,
            n_boots_est=args.n_boots_est,
            stability_selection=args.stability_selection,
            selection_frac=args.selection_frac,
            estimation_frac=args.estimation_frac,
            max_iter=10000,
            random_state=random_state)
        uoi.fit(X_new[train_idx, :], y[train_idx])
        uoi_coefs[fold] = uoi.coef_
        uoi_intercepts[fold] = uoi.intercept_
        uoi_scores[fold] = uoi.score(X_new[test_idx, :],
                                     y[test_idx])

    # store results in h5 file
    results = h5py.File(results_path, 'a')
    base = results.require_group(results_group)

    baseline_group = base.require_group('baseline')
    baseline_group['coefs'] = baseline_coefs
    baseline_group['intercepts'] = baseline_intercepts
    baseline_group['scores'] = baseline_scores

    uoi_group = base.require_group('uoi')
    uoi_group['coefs'] = uoi_coefs
    uoi_group['intercepts'] = uoi_intercepts
    uoi_group['scores'] = uoi_scores

    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--window_left', type=float, default=0.1)
    parser.add_argument('--window_right', type=float, default=0.)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=-1)
    parser.add_argument('--with_std', action='store_true')

    # fitter arguments
    parser.add_argument('--n_Cs', type=int, default=100)
    parser.add_argument('--cv', type=int, default=5)
    # UoI arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)

    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
