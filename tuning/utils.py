"""Utility functions for plotting tuning results."""
import h5py
import matplotlib.pyplot as plt
import numpy as np


def calculate_selection_ratio(coefs):
    """Calculate selection ratio for a set of coefficients."""
    n_max_coefs = coefs.shape[-1]
    selection_ratio = np.count_nonzero(coefs, axis=-1) / n_max_coefs
    return selection_ratio


def plot_metric_grid(baseline_group, fits_groups, metrics, fax=None):
    """Analyze a set of fits in a grid of subplots, using a user-provided
    set of metrics.

    Parameters
    ----------
    baseline_group : HDF5 Object
        The baseline algorithm to compare against.

    fits_groups : list of HDF5 objects
        A list of the coupling fits to look at.

    metrics : list of strings
        A list of the metrics to plot in the rows of the subplots.

    fax : tuple of (fig, axes) matplotlib objects
        If None, a (fig, axes) is created. Otherwise, fax are modified directly.

    Returns
    -------
    fax : tuple of (fig, axes) matplotlib objects
        The (fig, axes) on which the metrics were plotted.
    """
    n_algorithms = len(fits_groups)
    n_metrics = len(metrics)

    if fax is None:
        fig, axes = plt.subplots(n_metrics, n_algorithms,
                                 figsize=(3 * n_algorithms, 3 * n_metrics))
    else:
        fig, axes = fax

    # iterate over metrics
    for row_idx, metric in enumerate(metrics):
        if metric == 'selection_ratio':
            baseline_coefs = baseline_group['tuning_coefs'][:]
            baseline_selection_ratio = \
                calculate_selection_ratio(baseline_coefs).mean(axis=0)

        # iterate over algorithms
        for col_idx, algorithm in enumerate(fits_groups):
            if metric == 'selection_ratio':
                # calculate selection ratio for algorithm
                coefs = algorithm['tuning_coefs'][:]
                selection_ratio = calculate_selection_ratio(coefs).mean(axis=0)

                # plot direct comparison
                axes[row_idx, col_idx].scatter(
                    baseline_selection_ratio,
                    selection_ratio,
                    alpha=0.5,
                    color='k',
                    edgecolor='w')
            else:
                axes[row_idx, col_idx].scatter(
                    baseline_group[metric][:].mean(axis=0),
                    algorithm[metric][:].mean(axis=0),
                    alpha=0.5,
                    color='k',
                    edgecolor='w')

    return fig, axes


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
