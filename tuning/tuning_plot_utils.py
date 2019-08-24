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


def plot_retina_strf(filepath, cell_recording, fax=None, vmin=-1e-6, vmax=1e-6):
    """Plots a comparison of STRFs obtained on RET1 dataset using Lasso and
    UoI Lasso."""
    # create axes
    if fax is None:
        fig, axes = plt.subplots(1, 2, figsize=(3, 4))
    else:
        fig, axes = fax

    results = h5py.File(filepath, 'r')

    lasso_strf = results[cell_recording]['Lasso/strf'][:]
    uoi_strf = results[cell_recording]['UoI_AIC/strf'][:]

    axes[0].imshow(
        lasso_strf.T,
        cmap=plt.get_cmap('RdGy'),
        vmin=vmin, vmax=vmax)
    axes[0].set_aspect('auto')
    axes[0].tick_params(labelsize=10)
    axes[0].set_yticks(np.arange(lasso_strf.shape[0]))

    axes[1].imshow(
        uoi_strf.T,
        cmap=plt.get_cmap('RdGy'),
        vmin=vmin, vmax=vmax)
    axes[1].set_aspect('auto')
    axes[1].tick_params(labelsize=10)
    axes[1].set_yticks(np.arange(uoi_strf.shape[0]))

    axes[0].set_xticks([0, 24])
    axes[1].set_xticks([0, 24])
    axes[0].set_ylabel(r'\textbf{Position}', fontsize=15)
    axes[1].set_xlabel(r'\textbf{Time (s)}', fontsize=15)

    for ax in axes:
        ax.set_ylim([150, 250])

    axes[0].set_title(r'\textbf{Lasso}', fontsize=15)
    axes[1].set_title(r'\textbf{UoI}$_{\textbf{Lasso}}$', fontsize=15)

    return fig, axes
