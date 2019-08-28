"""Utility functions for plotting tuning results."""
import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_metric(
    results_path, metric, x='Lasso', y='UoI_Lasso_R2', fax=None, color='k',
    marker='o'
):
    """Compares a metric between two fitting procedures on a specified set of
    axes."""
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        fig, ax = fax

    results = h5py.File(results_path, 'r')
    x_data = results[x]
    y_data = results[y]

    # plot the metric
    if metric == 'selection_ratio':
        n_targets = x_data['tuning_coefs'].shape[-1] + 1

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
        x_plot, y_plot,
        alpha=0.5,
        color=color,
        edgecolor='w',
        marker=marker)

    results.close()

    return fig, ax


def plot_tuning_grid(fits_path, base, fax=None):
    """Plots the selection ratios and scores from the tuning fits."""
    # create plot
    if fax is None:
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    else:
        fig, axes = fax

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

    return fig, axes


def plot_retina_strf(
    results_path, cell_recording, fax=None, vmin=-1e-6, vmax=1e-6
):
    """Plots a comparison of STRFs obtained on RET1 dataset using Lasso and
    UoI Lasso. Does not include colorbar."""
    # create axes
    if fax is None:
        fig, axes = plt.subplots(1, 2, figsize=(3, 4))
    else:
        fig, axes = fax

    # extract fits
    results = h5py.File(results_path, 'r')
    lasso_strf = results[cell_recording]['Lasso/strf'][:]
    uoi_strf = results[cell_recording]['UoI_AIC/strf'][:]
    results.close()

    # plot lasso STRF
    axes[0].imshow(
        lasso_strf.T,
        cmap=plt.get_cmap('RdGy'),
        vmin=vmin, vmax=vmax)
    axes[0].set_aspect('auto')
    axes[0].tick_params(labelsize=10)
    axes[0].set_yticks(np.arange(lasso_strf.shape[0]))

    # plot UoI STRF
    axes[1].imshow(
        uoi_strf.T,
        cmap=plt.get_cmap('RdGy'),
        vmin=vmin, vmax=vmax)
    axes[1].set_aspect('auto')
    axes[1].tick_params(labelsize=10)
    axes[1].set_yticks(np.arange(uoi_strf.shape[0]))

    return fig, axes
