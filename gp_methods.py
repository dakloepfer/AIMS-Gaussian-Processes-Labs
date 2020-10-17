import numpy as np
import scipy
#TODO check this whole function with the true values given; check if Cholesky transform method does in the end avoid conditioning errors (check with source potentially)
def posterior_predictive_distr(X, y, target_noise, X_star, mean_f, cov_f, jitter=1e-5):
    '''
        Compute and returns the mean and covariance of the posterior predictive distribution at the m test points (dimension d) given known target values at n points (the known target values may have some variance)

        Args:
            X: points with known target values (n x d)
            y: known target values for corresponding point in X (n x 1)
            target_noise: variance (uncertainty) of the corresponding target value in y (n x 1)
            X_star: test points at which the posterior distribution should be evaluated (m x d)

            mean_f: mean function; function that computes the prior mean at a set of points (takes in n x d array, returns an n x 1 array)
            cov_f: covariance function; function that computes the prior covariance matrix between two sets of points (takes in one n x d and one m x d array, returns a n x m array) #TODO use lambda functions to turn a mean_f with parameters into a mean_f as required here

            jitter: small artificial noise that is added to the covariance function for computational stability (float)

        Returns:
            mean_star: mean of the posterior predictive distribution at the points determined by X_star (m x 1)
            cov_star: covariance matrix of the posterior predictive distribution at the points determined by X_star (m x m)
    '''

    # Compute number of points
    n = np.shape(y)[0]

    #print(target_noise * np.eye(n) + jitter * np.eye(n))
    # Cholesky decomposition
    L = np.linalg.cholesky(cov_f(X, X) + (target_noise + jitter) * np.eye(n))

    # Compute repeated terms
    k_star = cov_f(X_star, X) # prior covariance matrix at the test points with the data points as computed by cov_f

    # Compute mean_star
    helper_vec = scipy.linalg.solve_triangular(L, y - mean_f(X), lower=True) # intermediary helper vector to compute alpha
    alpha = scipy.linalg.solve_triangular(np.transpose(L), helper_vec)
    mean_star = mean_f(X_star) + (k_star @ alpha)

    # Compute cov_star
    v = scipy.linalg.solve_triangular(L, np.transpose(k_star), lower=True) # intermediary helper value
    cov_star = cov_f(X_star, X_star) - np.transpose(v) @ v

    return mean_star, cov_star


def neg_marg_log_likelihood(params, n_mean_f_params, mean_f, cov_f, X, y, target_noise, jitter=1e-5):
    '''
        Compute and returns the negative marginal log-likelihood given known target values at n points (the known target values may have some variance).

        Args:
            params: an array of numbers giving the values for the hyperparameters for mean_f and cov_f to be used, with the first n_mean_f_params parameters belonging to mean_f 
            n_mean_f_params: an integer giving the number of hyperparameters that mean_f requires.

            mean_f: mean function; function that computes the prior mean at a set of points (takes in n x d array and n_mean_f_params numerical hyperparameters, returns an n x 1 array)
            cov_f: covariance function; function that computes the prior covariance matrix between two sets of points (takes in one n x d and one m x d array and a series of numerical hyperparameters, returns an n x m array)

            X: points with known target values (n x d)
            y: known target values for corresponding points in X (n x 1)
            target_noise: variance (uncertainty) of the corresponding target value in y (n x 1)

            jitter: small artificial noise that is added to the covariance function for computational stability (float)

        Returns:
            neg_log_likelihood: a float giving the value for the negative marginal log-likelihood
    '''

    # find parameters for mean_f and cov_f
    mean_f_params = params[:n_mean_f_params]
    cov_f_params = params[n_mean_f_params:]

    # Compute number of points
    n = np.shape(y)[0]

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_f(X, X, cov_f_params) + (target_noise + jitter) * np.eye(n))
    
    helper_vec = scipy.linalg.solve_triangular(L, y - mean_f(X, mean_f_params), lower=True) # intermediary helper vector to compute alpha
    alpha = scipy.linalg.solve_triangular(np.transpose(L), helper_vec) # another intermediary helper value

    neg_log_likelihood = 0.5 * (np.transpose(y - mean_f(X, mean_f_params)) @ alpha) + np.sum(np.log(np.diagonal(L))) + 0.5 * n * np.log(2*np.pi)

    return np.squeeze(neg_log_likelihood)

    
def pred_neg_log_likelihood(true_targets, mean, cov, target_noise=0, jitter=1e-5):
    '''
        Computes the negative predictive log-likelihood, the negative log of the probability of sampling the true_targets from the posterior predictive distribution as computed by posterior_predictive_distr. This is essentially the same function as marg_log_likelihood, but it does not compute and with a factor of -1

        Args:
            true_targets: m x 1 array of the true targets

            mean: m x 1 array; the mean of the posterior predictive distribution evaluated at the points for which true_targets are provided.
            cov: m x m array; the covariance matrix obtained by evaluating the covariance function of the posterior predictive distribution between the points for which true_targets are provided.

            target_noise: variance due to uncertainty of the values given in true_targets
            jitter: float; artificial noise added to the covariance matrix for computational stability.

        Returns: 
            neg_log_likelihood: the negative log likelihood; the negative log of p(true_values | posterior predictive distribution)
    '''

    # Compute number of points
    n = np.shape(true_targets)[0]

    # Cholesky decomposition
    L = np.linalg.cholesky(cov + (target_noise + jitter) * np.eye(n))

    helper_vec = scipy.linalg.solve_triangular(L, true_targets - mean, lower=True) # intermediary helper vector to compute alpha
    alpha = scipy.linalg.solve_triangular(np.transpose(L), helper_vec) # another intermediary helper value

    neg_log_likelihood = 0.5 * (np.transpose(true_targets - mean) @ alpha) + np.sum(np.log(np.diagonal(L))) + 0.5 * n * np.log(2*np.pi)

    return np.squeeze(neg_log_likelihood)
