import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls
from sklearn.utils import check_random_state


def plot_ecog_bases(components, ecog, vmax=None, fax=None):
    n_components, n_electrodes = components.shape

    # set up figure axes
    if fax is None:
        n_rows = int(np.ceil(n_components / 3))
        fig, axes = plt.subplots(n_rows, 3, figsize=(16, n_rows * 3))
    else:
        fig, axes = fax

    for component_idx, component in enumerate(components):
        # extract current axis
        ax = axes.ravel()[component_idx]

        # reshape bases into electrode grid
        grid = np.zeros((8, 16))
        for electrode_idx in range(n_electrodes):
            x, y = ecog.get_xy_for_electrode(electrode_idx)
            grid[x, y] = component[electrode_idx]

        if vmax is None:
            vmax = np.max(components)

        ax.imshow(np.flip(grid, axis=0), cmap=plt.get_cmap('Greys'),
                  vmin=0, vmax=vmax)
        ax.axis('off')

    return fig, axes


def bi_cross_validator(
    X, nmf, ranks, row_frac=0.75, col_frac=0.75, n_reps=10, random_state=None
):
    rng = check_random_state(random_state)

    n_samples, n_features = X.shape
    errors = np.zeros(ranks.size)

    for rep in range(n_reps):
        print(rep)
        for idx, k in enumerate(ranks):
            rows = np.sort(rng.choice(n_samples,
                                      size=int(row_frac * n_samples),
                                      replace=False))
            cols = np.sort(rng.choice(n_features,
                                      size=int(col_frac * n_features),
                                      replace=False))

            X_negI_negJ = np.delete(np.delete(X, rows, axis=0), cols, axis=1)
            nmf.set_params(n_components=k)
            nmf.fit(X_negI_negJ)

            H_negI_negJ = nmf.components_
            W_negI_negJ = nmf.transform(X_negI_negJ)

            # side blocks
            X_I_negJ = np.delete(X, cols, axis=1)[rows]
            X_negI_J = np.delete(X, rows, axis=0)[:, cols]

            # fit coefficients in last block
            W_IJ = np.zeros((rows.size, k))
            H_IJ = np.zeros((k, cols.size))

            for row in range(W_IJ.shape[0]):
                W_IJ[row] = nnls(H_negI_negJ.T, X_I_negJ[row])[0]

            for col in range(H_IJ.shape[1]):
                H_IJ[:, col] = nnls(W_negI_negJ, X_negI_J[:, col])[0]

            X_IJ = X[rows][:, cols]
            X_IJ_hat = np.dot(W_IJ, H_IJ)
            errors[idx] += np.sum((X_IJ - X_IJ_hat)**2)

    k_max = ranks[np.argmin(errors)]
    return k_max, errors
