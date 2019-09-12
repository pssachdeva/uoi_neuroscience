import argparse
import h5py
import numpy as np

from neuropacks import NHP
from utils import apply_kalman_filter, apply_linear_decoder


def main(args):
    data_path = args.data_path
    results_path = args.results_path
    dt = args.bin_width

    left_idx = args.left_idx
    if args.right_idx == 0:
        right_idx = None
    else:
        right_idx = args.right_idx

    results = h5py.File(results_path, 'a')
    nhp = NHP(data_path=data_path)

    # extract neural responses
    Y = results['data/Y'][:]

    positions = nhp.get_binned_positions(bin_width=dt)
    positions = positions[left_idx:right_idx, :]
    x = positions[:, 0]
    y = positions[:, 1]

    # create k array
    max_ks = np.arange(args.min_max_k, args.max_max_k, args.max_k_spacing)
    n_max_ks = max_ks.size

    reps = args.reps

    _, _, _, base_corrs = apply_kalman_filter(
        x, y, Y, score=True, train_frac=args.train_frac
    )

    uoi_kalman = np.zeros((reps, n_max_ks, 2))
    css_kalman = np.zeros((reps, n_max_ks, 2))
    uoi_linear = np.zeros((reps, n_max_ks, 2))
    css_linear = np.zeros((reps, n_max_ks, 2))

    # iterate over repetitions
    for rep in range(reps):
        if args.verbose:
            print('Repetition ', str(rep))
        # iterate over ranks
        for k_idx, max_k in enumerate(max_ks):
            uoi_c = results['uoi/columns/%s/%s' % (rep, max_k)][:]
            _, _, _, corrs = apply_kalman_filter(
                x, y, Y[:, uoi_c], score=True, train_frac=args.train_frac
            )
            uoi_kalman[rep, k_idx, 0] = corrs[0]
            uoi_kalman[rep, k_idx, 1] = corrs[1]

            _, _, _, corrs = apply_linear_decoder(
                x, y, Y[:, uoi_c], score=True, train_frac=args.train_frac
            )
            uoi_linear[rep, k_idx, 0] = corrs[0]
            uoi_linear[rep, k_idx, 1] = corrs[1]

            css_c = results['css/columns/%s/%s' % (rep, max_k)][:]
            _, _, _, corrs = apply_kalman_filter(
                x, y, Y[:, css_c], score=True, train_frac=args.train_frac
            )
            css_kalman[rep, k_idx, 0] = corrs[0]
            css_kalman[rep, k_idx, 1] = corrs[1]

            _, _, _, corrs = apply_linear_decoder(
                x, y, Y[:, css_c], score=True, train_frac=args.train_frac
            )
            css_linear[rep, k_idx, 0] = corrs[0]
            css_linear[rep, k_idx, 1] = corrs[1]

        results['uoi/columns/%s' % rep]['kalman_scores'] = uoi_kalman
        results['css/columns/%s' % rep]['kalman_scores'] = css_kalman
        results['uoi/columns/%s' % rep]['linear_scores'] = uoi_linear
        results['css/columns/%s' % rep]['linear_scores'] = css_linear

    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--bin_width', type=float, default=0.25)
    parser.add_argument('--left_idx', type=int, default=0)
    parser.add_argument('--right_idx', type=int, default=0)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--min_max_k', type=int, default=2)
    parser.add_argument('--max_max_k', type=int, default=100)
    parser.add_argument('--max_k_spacing', type=int, default=2)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
