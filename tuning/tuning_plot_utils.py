import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_tuning_grid(fits_path, base, axes=None):
    if axes is None:
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))

    fits = h5py.File(fits_path, 'r')

    # load different fits
    lasso = fits[base + 'Lasso']
    uoi_r2 = fits[base + 'UoI_Lasso_R2']
    uoi_aic = fits[base + 'UoI_Lasso_AIC']
    uoi_bic = fits[base + 'UoI_Lasso_BIC']
    uois = [uoi_r2, uoi_aic, uoi_bic]

    n_targets = lasso['tuning_coefs'].shape[-1]

    for idx, algorithm in enumerate(uois):
        # first column: selection ratios
        lasso_selection_ratio = np.mean(np.count_nonzero(
            lasso['tuning_coefs'][:], axis=2
        )/n_targets, axis=0)
        algorithm_selection_ratio = np.mean(np.count_nonzero(
            algorithm['tuning_coefs'][:], axis=2
        )/n_targets, axis=0)
        axes[0, idx].scatter(
            lasso_selection_ratio,
            algorithm_selection_ratio,
            alpha=0.5,
            color='k',
            edgecolor='w'
        )

        # second column: explained variance
        axes[1, idx].scatter(
            np.mean(lasso['r2'][:], axis=0),
            np.mean(algorithm['r2'][:], axis=0),
            alpha=0.5,
            color='k',
            edgecolor='w'
        )

        # third column: Akaike information criterion
        axes[2, idx].scatter(
            np.mean(lasso['AIC'][:], axis=0),
            np.mean(algorithm['AIC'][:], axis=0),
            alpha=0.5,
            color='k',
            edgecolor='w'
        )

        # fourth column: Bayesian information criterion
        axes[3, idx].scatter(
            np.mean(lasso['BIC'][:], axis=0),
            np.mean(algorithm['BIC'][:], axis=0),
            alpha=0.5,
            color='k',
            edgecolor='w'
        )

    return axes
