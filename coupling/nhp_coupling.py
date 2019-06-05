"""
Performs a cross-validated coupling fit on spiking data from the PVC11 CRCNS
database. This dataset consists of primary visual cortex spiking responses to
gratings.

This script performs and stores coupling models on this dataset using a
desired fitting procedure.
"""
import argparse
import h5py
import numpy as np

from neuropacks import NHP
from sem import SEMSolver


def main(args):
    # check random state
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    # create data extraction object
    nhp = NHP(data_path=args.data_path)

    # get response matrix
    Y = nhp.get_response_matrix(
        bin_width=args.bin_width,
        region=args.region,
        transform=args.transform
    )
    # clear out empty units
    Y = Y[:, np.argwhere(Y.sum(axis=0) > 0).ravel()]

    # create solver object
    solver = SEMSolver(Y=Y)

    # create args
    args = {
        'model': 'c',
        'method': args.method,
        'targets': [0],
        'n_folds': args.n_folds,
        'random_state': random_state,
        'verbose': args.verbose,
        'fit_intercept': True,
        'max_iter': args.max_iter,
        'metrics': ['r2', 'AIC', 'BIC']
    }

    if args.method == 'Lasso':
        args['normalize'] = args.normalize
        args['cv'] = args.cv
    elif 'UoI' in args.method:
        # important: we use normalize/standardize to mean the same thing
        args['standardize'] = args.normalize
        args['n_boots_sel'] = args.n_boots_sel
        args['n_boots_est'] = args.n_boots_est
        args['selection_frac'] = args.selection_frac
        args['estimation_frac'] = args.estimation_frac
        args['n_lambdas'] = args.n_lambdas
        args['stability_selection'] = args.stability_selection
        args['estimation_score'] = args.estimation_score
    else:
        raise ValueError('Method is not valid.')

    if args.method == 'UoI_Poisson':
        args['solver'] = args.solver
        args['metrics'] = ['AIC', 'BIC']

    # perform cross-validated coupling fits
    results = solver.estimation(**args)

    results_file = h5py.File(args.results_path, 'a')
    group = results_file.create_group(args.results_group)
    # place results in group
    for key in results.keys():
        # need to handle training and test folds separately
        if key == 'training_folds' or key == 'test_folds':
            folds_group = group.create_group(key)
            for fold_key, fold_val in results[key].items():
                folds_group[fold_key] = fold_val
        else:
            group[key] = results[key]

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')

    # NHP arguments
    parser.add_argument('--region', default='M1')
    parser.add_argument('--bin_width', type=float, default=0.5)
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--transform', default='square_root')

    # fitter object arguments
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--random_state', type=int, default=-1)

    # LassoCV
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=5000)

    # UoI arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--estimation_score', default='r2')

    # UoI Poisson arguments
    parser.add_argument('--solver', default='lbfgs')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    main(args)
