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
