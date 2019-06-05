import argparse
import h5py
import numpy as np

from neuropacks import NHP
from pyuoi.decomposition import UoI_CUR, CUR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def main(args):
    nhp = NHP(data_path=args.data_path)

    # extract neural responses
    Y = nhp.get_response_matrix(
        bin_width=args.bin_width,
        region=args.region,
        transform=None
    )
    nonzero_idx = np.argwhere(Y.sum(axis=0) > 0).ravel()
    Y = Y[:, nonzero_idx]
    n_samples, n_features = Y.shape

    # get cursor position
    cursor_position = nhp.get_binned_positions(bin_width=args.bin_width)
    cursor_position_x = cursor_position[:, 0]
    cursor_position_y = cursor_position[:, 1]

    # create k array
    max_ks = np.arange(args.min_max_k, args.max_max_k, args.max_k_spacing)
    n_max_ks = max_ks.size

    n_splits = args.n_splits
    reps = args.reps

    # create storage for results
    uoi_reconstructions = np.zeros((reps, n_max_ks))
    cur_reconstructions = np.zeros((reps, n_max_ks))
    uoi_decoding_x = np.zeros((reps, n_max_ks, n_splits))
    uoi_decoding_y = np.zeros((reps, n_max_ks, n_splits))
    cur_decoding_x = np.zeros(uoi_decoding_x.shape)
    cur_decoding_y = np.zeros(uoi_decoding_y.shape)

    # iterate over ranks
    for rep in range(reps):
        print(rep)
        for k_idx, max_k in enumerate(max_ks):
            # perform UoI CSS
            uoi_cur = UoI_CUR(
                n_boots=args.n_boots,
                max_k=max_k,
                boots_frac=args.boots_frac
            )

            uoi_cur.fit(Y, ks=int(max_k))

            uoi_columns = uoi_cur.column_indices_
            n_columns = uoi_columns.size

            # perform ordinary CSS
            cur = CUR(max_k=max_k)
            cur.fit(Y)
            cur_columns = np.sort(cur.column_indices_[:n_columns])

            # extract selected columns
            Y_uoi = Y[:, uoi_columns]
            Y_cur = Y[:, cur_columns]

            # calculate reconstruction errors
            uoi_reconstruction = Y - np.dot(Y_uoi, np.dot(np.linalg.pinv(Y_uoi), Y))
            cur_reconstruction = Y - np.dot(Y_cur, np.dot(np.linalg.pinv(Y_cur), Y))
            uoi_reconstructions[rep, k_idx] = \
                np.sum(np.abs(uoi_reconstruction)) / Y.size
            cur_reconstructions[rep, k_idx] = \
                np.sum(np.abs(cur_reconstruction)) / Y.size

            # calculate decoding errors on position
            kf = KFold(n_splits=n_splits)
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Y)):
                # training and test sets
                Y_train, pos_train_x, pos_train_y = \
                    Y[train_idx], cursor_position_x[train_idx], \
                    cursor_position_y[train_idx]
                Y_test, pos_test_x, pos_test_y = \
                    Y[test_idx], cursor_position_x[test_idx], \
                    cursor_position_y[test_idx]

                # decode with uoi columns
                ols = LinearRegression()
                ols.fit(Y_train[:, uoi_columns], pos_train_x)
                uoi_decoding_x[rep, k_idx, fold_idx] = \
                    ols.score(Y_test[:, uoi_columns], pos_test_x)

                ols = LinearRegression()
                ols.fit(Y_train[:, uoi_columns], pos_train_y)
                uoi_decoding_y[rep, k_idx, fold_idx] = \
                    ols.score(Y_test[:, uoi_columns], pos_test_y)

                # decode with cur columns
                ols = LinearRegression()
                ols.fit(Y_train[:, cur_columns], pos_train_x)
                cur_decoding_x[rep, k_idx, fold_idx] = \
                    ols.score(Y_test[:, cur_columns], pos_test_x)

                ols = LinearRegression()
                ols.fit(Y_train[:, cur_columns], pos_train_y)
                cur_decoding_y[rep, k_idx, fold_idx] = \
                    ols.score(Y_test[:, cur_columns], pos_test_y)

    # save results
    results = h5py.File(args.results_path, 'a')
    results['data/Y'] = Y
    results['uoi/reconstructions'] = uoi_reconstructions
    results['uoi/decoding_x'] = uoi_decoding_x
    results['uoi/decoding_y'] = uoi_decoding_y
    results['cur/reconstructions'] = cur_reconstructions
    results['cur/decoding_x'] = cur_decoding_x
    results['cur/decoding_y'] = cur_decoding_y
    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path')
    parser.add_argument('--results_path')
    parser.add_argument('--bin_width', type=float, default=0.20)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--region', default='M1')
    parser.add_argument('--min_max_k', type=int, default=1)
    parser.add_argument('--max_max_k', type=int, default=100)
    parser.add_argument('--max_k_spacing', type=int, default=2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--n_boots', type=int, default=20)
    parser.add_argument('--boots_frac', type=float, default=0.8)
    args = parser.parse_args()

    main(args)
