"""
Performs a STRF fitting from retinal spiking data (RET 1 on CRCNS).
"""
import argparse
import h5py

from neuropacks import RET1 as Retina


def main(args):
    # create data extraction object
    retina = Retina(
        data_path=args.data_path,
        random_path=args.random_path
    )

    # perform STRF fitting
    strf, intercepts, training_scores, test_scores = \
        retina.calculate_strf_for_neurons(
            method=args.method,
            recording_idx=args.recording_idx,
            window_length=args.window_length,
            cells=args.cell,
            test_frac=0.1,
            return_scores=True,
            verbose=args.verbose,

            # general options
            normalize=args.normalize,

            # Lasso specific
            cv=args.cv,
            max_iter=args.max_iter,

            # UoI Lasso specific
            n_boots_sel=args.n_boots_sel,
            n_boots_est=args.n_boots_est,
            selection_frac=args.selection_frac,
            estimation_frac=args.estimation_frac,
            n_lambdas=args.n_lambdas,
            stability_selection=args.stability_selection,
            estimation_score=args.estimation_score
        )

    results_file = h5py.File(args.results_path, 'a')
    cell_recording = 'cell%s_recording%s' % (args.cell, args.recording_idx)
    group = results_file.create_group(
        cell_recording + '/' + args.results_group
    )
    # place results in group
    group['strf'] = strf
    group['intercepts'] = intercepts
    group['r2s_training'] = training_scores[0]
    group['bics_training'] = training_scores[1]
    group['aics_training'] = training_scores[2]
    group['r2s_test'] = test_scores[0]
    group['bics_test'] = test_scores[1]
    group['aics_test'] = test_scores[2]

    results_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path')
    parser.add_argument('--random_path')
    parser.add_argument('--results_path')
    parser.add_argument('--results_group')
    parser.add_argument('--method')
    parser.add_argument('--cell', type=int)
    parser.add_argument('--recording_idx', type=int)
    parser.add_argument('--window_length', type=float, default=0.5)
    parser.add_argument('--verbose', action='store_true')
    # fitter object arguments
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--max_iter', type=int, default=5000)
    # UoI Lasso arguments
    parser.add_argument('--n_boots_sel', type=int, default=50)
    parser.add_argument('--n_boots_est', type=int, default=50)
    parser.add_argument('--selection_frac', type=float, default=0.8)
    parser.add_argument('--estimation_frac', type=float, default=0.8)
    parser.add_argument('--n_lambdas', type=int, default=48)
    parser.add_argument('--stability_selection', type=float, default=1.)
    parser.add_argument('--estimation_score', default='r2')

    args = parser.parse_args()

    main(args)
