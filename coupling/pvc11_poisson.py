import argparse
import h5py

from neuropacks import PVC11
from sem import SEMSolver


def main(args):
    if args.random_state == -1:
        random_state = None
    else:
        random_state = args.random_state

    # create data extraction object
    pvc = PVC11(data_path=args.data_path)
    # get response matrix
    Y = pvc.get_response_matrix(
        transform=None
    )
    # for stratification of folds
    class_labels = pvc.get_design_matrix(form='label')

    # create solver object
    solver = SEMSolver(Y=Y)

    # perform cross-validated coupling fits
    results = solver.estimation(
        model='c',
        method='UoI_Poisson',
        class_labels=class_labels,
        targets=None,
        n_folds=args.n_folds,
        random_state=random_state,
        metrics=['BIC', 'AIC'],
        verbose=args.verbose,

        # UoI Lasso specific
        n_boots_sel=args.n_boots_sel,
        n_boots_est=args.n_boots_est,
        selection_frac=args.selection_frac,
        estimation_frac=args.estimation_frac,
        n_lambdas=args.n_lambdas,
        estimation_score=args.estimation_score,
        solver=args.solver,
        max_iter=args.max_iter
    )

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

    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=2332)
    parser.add_argument('--verbose', action='store_true')
    # UoI Lasso arguments
    parser.add_argument('--n_boots_sel', type=int, default=30)
    parser.add_argument('--n_boots_est', type=int, default=30)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=50)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--estimation_score', default='log')
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--solver', default='lbfgs')

    args = parser.parse_args()

    main(args)
