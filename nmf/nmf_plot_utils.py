import matplotlib.pyplot as plt
import numpy as np


def plot_ecog_bases(components, ecog, fax=None):
    n_components, n_electrodes = components.shape

    # set up figure axes
    if fax is None:
        n_rows = n_components // 3
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

        ax.imshow(np.flip(grid, axis=0))
        ax.axis('off')

    return fig, axes