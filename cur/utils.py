import numpy as np

from neuropacks import NHP
from pykalman import KalmanFilter
from pyuoi.decomposition import UoI_CUR, CUR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def compare_UoI_CUR_reconstruction_NHP(
    data_path, bin_width=0.25, region='M1'
):
    nhp = NHP(data_path=data_path)
    Y = nhp.get_response_matrix(
        bin_width=bin_width, region=region, transform=None
    )
    nz_indices = np.argwhere(Y.sum(axis=0) > 0).ravel()
    Y = Y[:, nz_indices]

    ks = np.arange(1, Y.shape[1])
    n_ks = ks.size

    n_columns_uoi = np.zeros(n_ks)
    n_columns_cur = np.zeros(n_ks)
    reconstruction_uoi = np.zeros(n_ks)
    reconstruction_cur = np.zeros(n_ks)

    for k_idx, k in enumerate(ks):
        uoicur = UoI_CUR(n_boots=20, max_k=k, boots_frac=0.8)
        cur = CUR(max_k=k)

        uoicur.fit(Y, ks=int(k))
        cur.fit(Y, c=k + 20)

        # number of columns
        n_columns_uoi[k_idx] = uoicur.column_indices_.size
        n_columns_cur[k_idx] = cur.column_indices_.size

        # reconstruction cost
        Y_uoi = uoicur.components_
        Y_cur = Y[:, cur.column_indices_[:uoicur.column_indices_.size]]
        Y_uoi_err = Y - np.dot(Y_uoi, np.dot(np.linalg.pinv(Y_uoi), Y))
        Y_cur_err = Y - np.dot(Y_cur, np.dot(np.linalg.pinv(Y_cur), Y))
        reconstruction_uoi[k_idx] = np.sum(np.abs(Y_uoi_err)) / Y.size
        reconstruction_cur[k_idx] = np.sum(np.abs(Y_cur_err)) / Y.size

    return (n_columns_cur, reconstruction_cur), (n_columns_uoi, reconstruction_uoi)


def decoding_comparison_nhp(data_path, bin_width=0.25, region='M1', n_folds=5):
    nhp = NHP(data_path=data_path)
    Y = nhp.get_response_matrix(
        bin_width=bin_width, region=region, transform=None
    )
    nz_indices = np.argwhere(Y.sum(axis=0) > 0).ravel()
    Y = Y[:, nz_indices]

    positions = nhp.get_binned_positions(bin_width=bin_width)
    x_pos = positions[:, 0]
    y_pos = positions[:, 1]

    ks = np.arange(1, Y.shape[1])
    n_ks = ks.size

    n_columns_uoi = np.zeros(n_ks)
    n_columns_cur = np.zeros(n_ks)
    decoding_x_uoi = np.zeros((n_ks, n_folds))
    decoding_y_uoi = np.zeros((n_ks, n_folds))
    decoding_x_cur = np.zeros((n_ks, n_folds))
    decoding_y_cur = np.zeros((n_ks, n_folds))

    for k_idx, k in enumerate(ks):
        uoicur = UoI_CUR(n_boots=20, max_k=k, boots_frac=0.8)
        cur = CUR(max_k=k)

        uoicur.fit(Y, ks=int(k))
        cur.fit(Y, c=k + 20)

        # number of columns
        n_columns_uoi[k_idx] = uoicur.column_indices_.size
        n_columns_cur[k_idx] = cur.column_indices_.size

        # reconstruction cost
        Y_uoi = uoicur.components_
        Y_cur = Y[:, cur.column_indices_[:uoicur.column_indices_.size]]

        kf = KFold(n_splits=n_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Y)):
            Y_uoi_train, Y_cur_train, x_pos_train, y_pos_train = \
                Y_uoi[train_idx], Y_cur[train_idx], x_pos[train_idx], y_pos[train_idx]
            Y_uoi_test, Y_cur_test, x_pos_test, y_pos_test = \
                Y_uoi[test_idx], Y_cur[test_idx], x_pos[test_idx], y_pos[test_idx]

            # decode with uoi columns
            ols = LinearRegression()
            ols.fit(Y_uoi_train, x_pos_train)
            decoding_x_uoi[k_idx, fold_idx] = ols.score(Y_uoi_test, x_pos_test)

            ols = LinearRegression()
            ols.fit(Y_uoi_train, y_pos_train)
            decoding_y_uoi[k_idx, fold_idx] = ols.score(Y_uoi_test, y_pos_test)

            # decode with cur columns
            ols = LinearRegression()
            ols.fit(Y_cur_train, x_pos_train)
            decoding_x_cur[k_idx, fold_idx] = ols.score(Y_cur_test, x_pos_test)

            ols = LinearRegression()
            ols.fit(Y_cur_train, y_pos_train)
            decoding_y_cur[k_idx, fold_idx] = ols.score(Y_cur_test, y_pos_test)

    return (n_columns_cur, decoding_x_cur, decoding_y_cur), \
           (n_columns_uoi, decoding_x_uoi, decoding_y_uoi)


def apply_kalman_filter(x, y, Y, dt=0.25, train_frac=0.8):
    """Trains a Kalman Filter to incoming neural data, and applies it to test
    data."""
    n_total_samples = Y.shape[0]
    n_train_samples = int(n_total_samples * train_frac)

    vx = np.ediff1d(x) / dt
    vy = np.ediff1d(y) / dt

    X = np.vstack((x[:-1], y[:-1], vx, vy)).T

    X_train = X[:n_train_samples]
    X_test = X[n_train_samples:]
    Z_train = Y[:n_train_samples]
    Z_test = Y[n_train_samples:]

    # center input
    X_train -= X_train.mean(axis=0, keepdims=True)
    X_test -= X_train.mean(axis=0, keepdims=True)
    Z_train -= Z_train.mean(axis=0, keepdims=True)
    Z_test -= Z_test.mean(axis=0, keepdims=True)

    # standardize input
    Z_train /= Z_train.std(axis=0, keepdims=True)
    Z_test /= Z_train.std(axis=0, keepdims=True)

    # fill in the kinematics matrix
    A = np.identity(4)
    # ensure velocity explains positions
    A[0, 2] = dt
    A[1, 3] = dt
    # fit velocity
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_train[:-1, 2:4], X_train[1:, 2:4])
    A[2:4, 2:4] = ols.coef_

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=C,
        observation_offsets=d,
        transition_offsets=np.zeros(4),
        transition_covariance=W,
        observation_covariance=Q
    )