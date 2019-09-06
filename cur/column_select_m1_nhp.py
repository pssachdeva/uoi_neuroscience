import argparse
import h5py
import numpy as np

from neuropacks import NHP
from pyuoi.decomposition import UoI_CUR, CUR


def main(args):
    dt = args.bin_width

    nhp = NHP(data_path=args.data_path)

    # extract neural responses
    Y = nhp.get_response_matrix(
        bin_width=dt,
        region=args.region,
        transform='square_root')
    nonzero_idx = np.argwhere(Y.sum(axis=0) > 0).ravel()
    Y = Y[:, nonzero_idx]
    n_samples, n_features = Y.shape

    # create k array
    max_ks = np.arange(args.min_max_k, args.max_max_k, args.max_k_spacing)
    n_max_ks = max_ks.size

    reps = args.reps

    results = h5py.File(args.results_path, 'a')
    results['data/Y'] = Y
    uoi = results.create_group('uoi')
    css = results.create_group('css')

    # create storage for results
    uoi['reconstructions'] = np.zeros((reps, n_max_ks))
    css['reconstructions'] = np.zeros((reps, n_max_ks))

    # iterate over repetitions
    for rep in range(reps):
        if args.verbose:
            print('Repetition ', str(rep))
        # iterate over ranks
        for k_idx, max_k in enumerate(max_ks):
            # perform UoI CSS
            uoi_css = UoI_CUR(
                n_boots=args.n_boots,
                max_k=max_k,
                boots_frac=args.boots_frac)
            uoi_css.fit(Y, ks=int(max_k))
            uoi_columns = uoi_css.column_indices_
            n_columns = uoi_columns.size
            # perform ordinary CSS
            css_fit = CUR(max_k=max_k)
            css_fit.fit(Y)
            css_columns = np.sort(css_fit.column_indices_[:n_columns])

            # store column indices
            uoi['columns/' + str(rep) + '/' + str(max_k)] = uoi_columns
            css['columns/' + str(rep) + '/' + str(max_k)] = css_columns

            # extract selected columns
            Y_uoi = Y[:, uoi_columns]
            Y_css = Y[:, css_columns]

            # calculate reconstruction errors
            uoi_reconstruction = Y - np.dot(Y_uoi, np.dot(np.linalg.pinv(Y_uoi), Y))
            css_reconstruction = Y - np.dot(Y_css, np.dot(np.linalg.pinv(Y_css), Y))
            uoi['reconstructions'][rep, k_idx] = \
                np.sum(np.abs(uoi_reconstruction)) / Y.size
            css['reconstructions'][rep, k_idx] = \
                np.sum(np.abs(css_reconstruction)) / Y.size

    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--bin_width', type=float, default=0.25)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--region', default='M1')
    parser.add_argument('--min_max_k', type=int, default=2)
    parser.add_argument('--max_max_k', type=int, default=100)
    parser.add_argument('--max_k_spacing', type=int, default=2)
    parser.add_argument('--n_boots', type=int, default=20)
    parser.add_argument('--boots_frac', type=float, default=0.8)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
