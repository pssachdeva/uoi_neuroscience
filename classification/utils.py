import matplotlib.pyplot as plt
import numpy as np


def idx_to_xy(idx, dx, dy):
    x = idx % dx
    y = dy - 1 - (idx // dx)
    return x, y


def plot_basal_ganglia_coefs(
    baseline_coefs, uoi_coefs, vmin=-1.75, vmax=1.75, row_columns=(3, 6), fax=None
):
    if fax is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    else:
        fig, axes = fax

    baseline_coefs = np.median(baseline_coefs, axis=0)
    uoi_coefs = np.median(uoi_coefs, axis=0)

    baseline_img = axes[0].imshow(
        np.flip(baseline_coefs.reshape(row_columns), axis=0),
        vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap('RdGy'))

    for idx, coef in enumerate(baseline_coefs):
        if coef == 0:
            x, y = idx_to_xy(idx, row_columns[1], row_columns[0])
            axes[0].scatter(x, y, marker='x', color='k', s=225)

    uoi_img = axes[1].imshow(
        np.flip(uoi_coefs.reshape(row_columns), axis=0),
        vmin=vmin, vmax=vmax,
        cmap=plt.get_cmap('RdGy'))

    for idx, coef in enumerate(uoi_coefs):
        if coef == 0:
            x, y = idx_to_xy(idx, row_columns[1], row_columns[0])
            axes[1].scatter(x, y, marker='x', color='k', s=225)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes, baseline_img, uoi_img
