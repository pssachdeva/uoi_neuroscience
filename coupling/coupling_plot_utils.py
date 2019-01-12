import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_coupling_grid(fits_path, axes=None):
    if axes is None:
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    fits = h5py.File(fits_path, 'r')

    # load different fits
    lasso = fits['Lasso']
    uoi_r2 = fits['UoI_Lasso_R2']
    uoi_aic = fits['UoI_Lasso_AIC']
    uoi_bic = fits['UoI_Lasso_BIC']
    uois = [uoi_r2, uoi_aic, uoi_bic]

    n_targets = lasso['coupling_coefs'].shape[-1]

    for idx, algorithm in enumerate(uois):
        # first column: selection ratios
        lasso_selection_ratio = np.mean(np.count_nonzero(
            lasso['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)
        algorithm_selection_ratio = np.mean(np.count_nonzero(
            algorithm['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)
        axes[idx, 0].scatter(
            lasso_selection_ratio,
            algorithm_selection_ratio,
            alpha=0.75,
            color='k',
            edgecolor='w'
        )

        # selection ratio must be between 0 and 1
        axes[idx, 0].set_xlim([0, 1])
        axes[idx, 0].set_ylim([0, 1])
        axes[idx, 0].set_aspect('equal')

        # second column: explained variance
        axes[idx, 1].scatter(
            np.mean(lasso['r2'][:], axis=0),
            np.mean(algorithm['r2'][:], axis=0),
            alpha=0.75,
            color='k',
            edgecolor='w'
        )

        # explained variance must be between 0 and 1
        axes[idx, 1].set_xlim([0, 1])
        axes[idx, 1].set_ylim([0, 1])
        axes[idx, 1].set_aspect('equal')

        # third column: Akaike information criterion
        axes[idx, 2].scatter(
            np.mean(lasso['AIC'][:], axis=0),
            np.mean(algorithm['AIC'][:], axis=0),
            alpha=0.75,
            color='k',
            edgecolor='w'
        )

        # fourth column: Bayesian information criterion
        axes[idx, 3].scatter(
            np.mean(lasso['BIC'][:], axis=0),
            np.mean(algorithm['BIC'][:], axis=0),
            alpha=0.75,
            color='k',
            edgecolor='w'
        )

    # row labels
    axes[0, 0].set_ylabel(r'\textbf{UoI, $R^2$}')
    axes[1, 0].set_ylabel(r'\textbf{UoI, AIC}')
    axes[2, 0].set_ylabel(r'\textbf{UoI, BIC}')

    # column labels
    axes[0, 0].set_title(
        r'\textbf{Selection}' '\n' r'\textbf{Ratio}', fontsize=21
    )
    axes[0, 1].set_title(
        r'\textbf{Explained}' '\n' r'\textbf{Variance}', fontsize=21
    )
    axes[0, 2].set_title(r'\textbf{AIC}', fontsize=21)
    axes[0, 3].set_title(r'\textbf{BIC}', fontsize=21)

    return axes
