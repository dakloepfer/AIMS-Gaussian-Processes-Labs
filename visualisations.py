'''File for scripts GP visualisations'''

import numpy as np
import matplotlib.pyplot as plt

def plot_true_data(x, y, x_label, y_label, y_noise=0):
    '''
        Plots the data given by x and y using '+' with errorbars and adds labels and a legend.

        Args:
            x: n x 1 array of x-values
            y: n x 1 array of y-values
            x_label: string; gives the label for the x-axis
            y_label: string; gives the label for the y-axis
            y_noise: n x 1 array or float (if equal noise for all values) of the standard deviations of the y-values; default is no noise 

        Returns:
            None
    '''
    if np.isscalar(y_noise):
        y_noise = y_noise * np.ones_like(y)

    plt.errorbar(x, y, yerr=y_noise, fmt='+', label='ground truths', color='black', zorder=0)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_mean(x, mean, x_label='', y_label=''):
    '''
        Plots the mean prediction of a Gaussian Process by interpolating the values in x and y.

        Args:
            x: n x 1 array of x-values (should be quite dense)
            mean: n x 1 array of predicted mean values
            x_label: optional string, label for the x-axis
            y_label: optional string, label for the y-axis

        Returns:
            None
    '''

    plt.plot(x, mean, label='mean', color='red', zorder=10)

    plt.legend()

    if len(x_label) > 0:
        plt.xlabel(x_label)
    if len(y_label) > 0:
        plt.ylabel(y_label)

def plot_uncertainty(x, mean, cov_matrix, n_stdevs=1, x_label='', y_label=''):
    '''
        Plots the uncertainty in the predicted mean by filling in the area a number of standard deviations above and below the mean.

        Args:
            x: n x 1 array of x-values (should be quite dense)
            mean: n x 1 array of predicted mean values
            cov_matrix: n x n array of the posterior covariance matrix of the points
            n_stdevs: integer or list of integers; function either fills in only the single band given by a single integer or all the bands given in the list with varying shadings
            x_label: optional string, label for the x-axis
            y_label: optional string, label for the y-axis

        Returns: 
            None
    '''

    stdevs = np.sqrt(np.diagonal(cov_matrix))

    if np.isscalar(n_stdevs):
        n_stdevs = [n_stdevs]

    for n in n_stdevs:
        plt.fill_between(x, mean + n*stdevs, mean - n*stdevs, label='%d stdevs' %n, zorder=-10*n, color='red', alpha=1/(n+2), edgecolor='none')

    plt.legend()