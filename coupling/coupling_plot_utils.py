import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wilcoxon


def get_dataset(fits_path, metric, key):
    fits = h5py.File(fits_path, 'r')
    data = fits[key]

    n_targets = data['coupling_coefs'].shape[-1]

    if metric == 'selection_ratio':
        x = np.mean(np.count_nonzero(
            data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)

    elif metric == 'r2' or metric == 'BIC':
        x = np.mean(data[metric][:], axis=0)
    else:
        raise ValueError('Metric not available.')

    return x


def wilcoxon_coupling(fits_path, metric, x='Lasso', y='UoI_Lasso_R2'):
    fits = h5py.File(fits_path, 'r')
    x_data = fits[x]
    y_data = fits[y]

    n_targets = x_data['coupling_coefs'].shape[-1]

    if metric == 'selection_ratio':
        x = np.mean(np.count_nonzero(
            x_data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)
        y = np.mean(np.count_nonzero(
            y_data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)

    elif metric == 'r2' or metric == 'BIC':
        x = np.mean(x_data[metric][:], axis=0)
        y = np.mean(y_data[metric][:], axis=0)
    else:
        raise ValueError('Metric not available.')

    return wilcoxon(x, y)


def plot_metric(
    fits_path, metric, x='Lasso', y='UoI_Lasso_R2', ax=None, color='k',
    marker='o'
):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(2, 2))

    fits = h5py.File(fits_path, 'r')
    x_data = fits[x]
    y_data = fits[y]

    n_targets = x_data['coupling_coefs'].shape[-1]

    if metric == 'selection_ratio':
        x_plot = np.mean(np.count_nonzero(
            x_data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)
        y_plot = np.mean(np.count_nonzero(
            y_data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)

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


def plot_coupling_grid(fits_path, axes=None):
    if axes is None:
        fig, axes = plt.subplots(4, 3, figsize=(9, 12))

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
