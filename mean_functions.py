'''
    A file to collect different mean functions for use in a Gaussian Process.
    Each function computes the prior mean at a set of points, taking in one n x d array and an array of numerical hyperparameters and returning an n x 1 array.
'''

import numpy as np


def const(X, params):
    '''
        Function returns a constant mean as specified by the one parameter in params.

        Args:
            X: n x d array

            params: 1 x 1 array containing the constant value that the function returns.

        Returns:
            means: n x 1 array containing the constant parameter.
    '''

    return params[0] * np.ones([np.shape(X)[0], 1])