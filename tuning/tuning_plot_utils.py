import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_metric(
    fits_path, metric, x='Lasso', y='UoI_Lasso_R2', ax=None, color='k',
    marker='o'
):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(2, 2))

    fits = h5py.File(fits_path, 'r')
    x_data = fits[x]
    y_data = fits[y]

    n_targets = x_data['tuning_coefs'].shape[-1] + 1

    if metric == 'selection_ratio':
        x_plot = np.mean((
            np.count_nonzero(x_data['tuning_coefs'][:], axis=2) + 1)/n_targets,
            axis=0
        )
        y_plot = np.mean((
            np.count_nonzero(y_data['tuning_coefs'][:], axis=2) + 1)/n_targets,
            axis=0
        )

    elif metric == 'r2' or metric == 'BIC':
        x_plot = np.mean(x_data[metric][:], axis=0)
        y_plot = np.mean(y_data[metric][:], axis=0)

    else:
        raise ValueError('Metric not available.')

    ax.scatter(
        x_plot,
        y_plot,
        alpha=0.5,
        color=color,
        edgecolor='w',
        marker=marker
    )

    return ax



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
