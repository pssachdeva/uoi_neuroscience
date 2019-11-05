import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from pyuoi.linear_model import UoI_Lasso, UoI_Poisson
from scipy.stats import hypergeom
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression, LassoCV


def get_fitter(method, **kwargs):
    """Returns an object to perform a regression fit.

    Parameters
    ----------
    method : string
        The method to use when performing the fit. Can be one of:
            'OLS' (Ordinary Least Squares)
            'Lasso' (Cross-validated Lasso)
            'UoI_Lasso' (UoI Lasso).

    kwargs :
        Additional arguments to pass to each constructor.

    Returns
    -------
    fitter : object
        The object that can perform a regression fit using the fit()
        function.
    """
    # create fitting object
    if method == 'OLS':
        fitter = LinearRegression(
            fit_intercept=kwargs.get('fit_intercept', True),
            normalize=kwargs.get('normalize', True)
        )

    elif method == 'Lasso':
        fitter = LassoCV(
            normalize=kwargs.get('normalize', True),
            cv=kwargs.get('cv', 10),
            max_iter=kwargs.get('max_iter', 5000)
        )

    elif method == 'UoI_Lasso':
        fitter = UoI_Lasso(
            n_boots_sel=kwargs.get('n_boots_sel', 30),
            n_boots_est=kwargs.get('n_boots_est', 30),
            selection_frac=kwargs.get('selection_frac', 0.8),
            estimation_frac=kwargs.get('estimation_frac', 0.8),
            estimation_score=kwargs.get('estimation_score', 'r2'),
            stability_selection=kwargs.get('stability_selection', 1.0),
            standardize=kwargs.get('standardize', True),
            fit_intercept=True,
            max_iter=kwargs.get('max_iter', 5000)
        )

    elif method == 'UoI_Poisson':
        fitter = UoI_Poisson(
            n_lambdas=kwargs.get('n_lambdas', 50),
            n_boots_sel=kwargs.get('n_boots_sel', 30),
            n_boots_est=kwargs.get('n_boots_est', 30),
            selection_frac=kwargs.get('selection_frac', 0.8),
            estimation_frac=kwargs.get('estimation_frac', 0.8),
            stability_selection=kwargs.get('stability_selection', 1.0),
            estimation_score=kwargs.get('estimation_score', 'log'),
            solver=kwargs.get('solver', 'lbfgs'),
            standardize=kwargs.get('standardize', True),
            fit_intercept=True,
            max_iter=kwargs.get('max_iter', 10000),
            warm_start=False
        )
    else:
        raise ValueError("Incorrect method specified.")

    return fitter


def create_graph(fits_path, dataset, weighted=False, directed=False):
    fits = h5py.File(fits_path, 'r')
    coefs = np.median(fits[dataset]['coupling_coefs'][:], axis=0)
    n_neurons = coefs.shape[0]
    A = np.zeros((n_neurons, n_neurons))

    # create adjacency matrix
    for neuron in range(n_neurons):
        A[neuron] = np.insert(coefs[neuron], neuron, 0)

    if not weighted:
        A = (A != 0).astype('int')

    if directed:
        G = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.DiGraph())
    else:
        A = (A + A.T) / 2
        G = nx.convert_matrix.from_numpy_matrix(A, create_using=nx.Graph())

    return G


def get_dataset(fits_path, metric, key):
    fits = h5py.File(fits_path, 'r')
    data = fits[key]

    n_targets = data['coupling_coefs'].shape[-1]

    if metric == 'selection_ratio':
        x = np.median(np.count_nonzero(
            data['coupling_coefs'][:], axis=2
        )/n_targets, axis=0)

    elif metric == 'r2' or metric == 'BIC':
        x = np.mean(data[metric][:], axis=0)
    else:
        raise ValueError('Metric not available.')

    return x


def coupling_coef_corrs(fits_path, dataset1, dataset2):
    """Calculate the correlation coefficients between all sets of coupling fits
    for two different procedures."""
    fits = h5py.File(fits_path, 'r')
    coefs1 = np.median(fits[dataset1]['coupling_coefs'][:], axis=0)
    coefs2 = np.median(fits[dataset2]['coupling_coefs'][:], axis=0)

    n_neurons = coefs1.shape[0]
    corrs = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        corrs[neuron] = np.corrcoef(coefs1[neuron], coefs2[neuron])[0, 1]

    return corrs


def selection_profiles_by_chance(fits_path, dataset1, dataset2):
    """Calculate the probability that the selection profile of dataset2 would
    match up with the selection profile of dataset1 according to the
    hypergeometric distribution."""
    fits = h5py.File(fits_path, 'r')
    true = np.median(fits[dataset1]['coupling_coefs'][:], axis=0)
    compare = np.median(fits[dataset2]['coupling_coefs'][:], axis=0)

    n_neurons, M = true.shape
    probabilities = np.zeros(n_neurons)

    for neuron in range(n_neurons):
        n = np.count_nonzero(true[neuron])
        N = np.count_nonzero(compare[neuron])
        rv = hypergeom(M=M, n=n, N=N)

        overlap = np.count_nonzero(true[neuron] * compare[neuron])
        probabilities[neuron] = 1 - rv.cdf(x=overlap)

    return probabilities


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


def calculate_selection_ratio(coefs):
    """Calculate selection ratio for a set of coefficients."""
    n_max_coefs = coefs.shape[-1]
    selection_ratio = np.count_nonzero(coefs, axis=-1) / n_max_coefs
    return selection_ratio


def plot_coupling_grid(baseline_group, fits_groups, metrics, fax=None):
    """Analyze a set of coupling fits in a grid of subplots, using a user-provided
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
            baseline_coefs = baseline_group['coupling_coefs'][:]
            baseline_selection_ratio = \
                calculate_selection_ratio(baseline_coefs).mean(axis=0)

        # iterate over algorithms
        for col_idx, algorithm in enumerate(fits_groups):
            if metric == 'selection_ratio':
                # calculate selection ratio for algorithm
                coefs = algorithm['coupling_coefs'][:]
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


def log_likelihood(y_true, y_pred):
    """Calculates the log-likelihood for a Poisson model.

    Parameters
    ----------
    y_true : ndarray
        The true response values.

    y_pred : ndarray
        The predicted response values, according to the Poisson model.

    Returns
    -------
    ll : float
        The total log-likelihood of the data.
    """
    ll = np.sum(y_true * np.log(y_pred) - y_pred)
    return ll


def deviance(y_true, y_pred):
    """Calculates the deviance for a Poisson model.

    Parameters
    ----------
    y_true : ndarray
        The true response values.

    y_pred : ndarray
        The predicted response values, according to the Poisson model.

    Returns
    -------
    dev : float
        The total deviance of the data.
    """
    # calculate log-likelihood of the predicted values
    ll_pred = log_likelihood(y_true, y_pred)
    # calculate log-likelihood of the true data
    y_true_nz = y_true[y_true != 0]
    ll_true = log_likelihood(y_true_nz, y_true_nz)
    # calculate deviance
    dev = ll_true - ll_pred
    return dev


def AIC(y_true, y_pred, n_features):
    """Calculates the AIC for a Poisson model.

    Parameters
    ----------
    y_true : ndarray
        The true response values.

    y_pred : ndarray
        The predicted response values, according to the Poisson model.

    n_features : int
        The number of non-zero features used in the model.

    Returns
    -------
    AIC : float
        The Akaike Information Criterion for the data.
    """
    ll = log_likelihood(y_true, y_pred)
    AIC = 2 * n_features - 2 * ll
    return AIC


def BIC(y_true, y_pred, n_features):
    """Calculates the BIC for a Poisson model.

    Parameters
    ----------
    y_true : ndarray
        The true response values.

    y_pred : ndarray
        The predicted response values, according to the Poisson model.

    n_features : int
        The number of non-zero features used in the model.

    Returns
    -------
    BIC : float
        The Bayesian Information Criterion for the data.
    """
    ll = log_likelihood(y_true, y_pred)
    n_samples = y_true.size
    BIC = np.log(n_samples) * n_features - 2 * ll
    return BIC
