import argparse
import h5py

from neuropacks import PVC11
from sem import SEMSolver


def main(args):
    # create data extraction object
    pvc = PVC11(data_path=args.data_path)
    # get response matrix
    Y = pvc.get_response_matrix(
        transform=args.transform
    )
    # for stratification of folds
    class_labels = pvc.get_design_matrix(form='label')

    # create solver object
    solver = SEMSolver(Y=Y)

    # perform cross-validated coupling fits
    results = solver.estimation(
        model='c',
        method=args.method,
        class_labels=class_labels,
        targets=None,
        n_folds=args.n_folds,
        metrics=['r2', 'BIC', 'AIC'],
        # general options
        normalize=args.normalize,
        # UoI Lasso specific
        estimation_score=args.estimation_score
    )

    results_file = h5py.File(args.results_path, 'w')
    group = results_file.create_group(args.results_group)
    # place results in group
    for key in results.keys():
        group[key] = results[key]

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')
    parser.add_argument('--n_folds', type=int)
    # optional arguments
    parser.add_argument('--transform', 'square_root')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--estimation_score', default='r2')

    args = parser.parse_args()
