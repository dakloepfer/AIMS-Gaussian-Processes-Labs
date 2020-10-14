'''
    A file to collect different covariance functions for use in a Gaussian Process.
    Each function computes the prior covariance matrix between two sets of points, taking in one n x d and one m x d array and a series of numerical hyperparameters and returning an n x m array.
'''

import numpy as np
from scipy.spatial import distance

def rbf(X, Y, params):
    '''
        Computes the covariance matrix between two sets of points using the radial basis function (exponentiated quadratic) kernel. The points have dimension d and the distance function used is the Euclidean distance.

        Args:
            X: n x d array
            Y: m x d array

            params: 2 x 1 array of two parameters, alpha and beta, for the radial basis function.


        Returns:
            cov_matrix: n x m array that is the computed covariance matrix
    '''

    alpha = params[0]
    beta = params[1]

    dists = distance.cdist(X, Y, metric='sqeuclidean')

    return alpha**2 * np.exp(-0.5 * dists / (beta**2)) # follow convention by including factor of 0.5


def periodic(X, Y, params):
    '''
        Computes the covariance matrix between two sets of points using a periodic kernel. The points have dimension d and the distance function used is the Euclidean distance.

        Args:
            X: n x d array
            Y: m x d array

            params: 3 x 1 array of two parameters, alpha, beta, and period, for the periodic kernel.


        Returns:
            cov_matrix: n x m array that is the computed covariance matrix
    '''

    alpha = params[0]
    beta = params[1]
    period = params[2]

    dists = distance.cdist(X, Y, metric='euclidean')

    return alpha**2 * np.exp(-2 * np.square(np.sin(np.pi * dists / period)) / (beta**2)) # follow convention by including factors of 2 and np.pi


def matern_half(X, Y, params):
    '''
        Computes the covariance matrix between two sets of points using a a Matérn kernel with nu = 1/2. The points have dimension d and the distance function used is the Euclidean distance.

        Args:
            X: n x d array
            Y: m x d array

            params: 2 x 1 array of two parameters, alpha and beta, for the Matérn-1/2 kernel.


        Returns:
            cov_matrix: n x m array that is the computed covariance matrix
    '''

    alpha = params[0]
    beta = params[1]

    dists = distance.cdist(X, Y, metric='euclidean')

    return alpha**2 * np.exp(-dists / beta)
