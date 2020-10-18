'''File for scripts GP visualisations'''

import numpy as np
import matplotlib.pyplot as plt

def plot_training_data(x, y, y_noise=0, x_label=None, y_label=None):
    '''
        Plots the data given by x and y using '+' with errorbars and adds labels and a legend.
        This function is to be used for the training data from which the model predicts values.

        Args:
            x: n x 1 array of x-values
            y: n x 1 array of y-values
            y_noise: n x 1 array or float (if equal noise for all values) of the standard deviations of the y-values; default is no noise 
            x_label: optional string; gives the label for the x-axis
            y_label: optional string; gives the label for the y-axis

        Returns:
            None
    '''

    plt.errorbar(x, y, yerr=y_noise, fmt='+', label='training data', color='black', zorder=0)

    plt.legend()

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

def plot_ground_truth(x, y, y_noise=0, x_label=None, y_label=None):
    '''
        Plots the data given by x and y using '+' with errorbars and adds labels and a legend.
        This function is to be used for the ground truth data for which the model predicts values.

        Args:
            x: n x 1 array of x-values
            y: n x 1 array of y-values
            x_label: optional string; gives the label for the x-axis
            y_label: optional string; gives the label for the y-axis
            y_noise: n x 1 array or float (if equal noise for all values) of the standard deviations of the y-values; default is no noise 

        Returns:
            None
    '''

    plt.errorbar(x, y, yerr=y_noise, fmt='+', label='ground truths', color='blue', zorder=0)

    plt.legend()

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)


def plot_mean(x, mean, x_label=None, y_label=None):
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

    plt.plot(x, mean, label='mean', color='red', zorder=10, linewidth=0.75)

    plt.legend()

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

def plot_uncertainty(x, mean, cov_matrix, n_stdevs=1, x_label=None, y_label=None):
    '''
        Plots the uncertainty in the predicted mean by filling in the area a number of standard deviations above and below the mean.

        Args:
            x: n x 1 array of x-values (should be quite dense)
            mean: n x 1 array of predicted mean values
            cov_matrix: n x n array of the posterior covariance matrix of the points or n x 1 vector of the standard deviations directly
            n_stdevs: integer or list of integers; function either fills in only the single band given by a single integer or all the bands given in the list with varying shadings
            x_label: optional string, label for the x-axis
            y_label: optional string, label for the y-axis

        Returns: 
            None
    '''

    if np.shape(cov_matrix)[0] == np.shape(cov_matrix)[1]:
        stdevs = np.sqrt(np.diagonal(cov_matrix))
    else:
        stdevs = np.squeeze(cov_matrix)

    if np.isscalar(n_stdevs):
        n_stdevs = [n_stdevs]

    for n in n_stdevs:

        upper_limit = np.squeeze(mean) + n*stdevs
        lower_limit = np.squeeze(mean) - n*stdevs
        
        plt.fill_between(np.squeeze(x), upper_limit, lower_limit, label='%d stdevs' %n, zorder=-10*n, color=(1, 1-1/(2*n-0.5), 1-1/(2*n-0.5)), edgecolor='none')

    plt.legend()

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)


def plot_function_draws(x, mean, cov_matrix, n_draws=3, x_label=None, y_label=None):
    '''
        Plots several function draws from the posterior distribution.

        Args:
            x: n x 1 array of x-values (should be quite dense); function will be evaluated at these points
            mean: n x 1 array of predicted mean values
            cov_matrix: n x n array of the posterior covariance matrix of the points
            n_draws: integer; number of function draws that should be plotted
            x_label: optional string, label for the x-axis
            y_label: optional string, label for the y-axis

        Returns: 
            None
    '''
    mean = np.squeeze(mean)

    for i in range(0, n_draws):

        y = np.random.default_rng().multivariate_normal(mean, cov_matrix)
        plt.plot(x, y, label='sampled function %d' %(i+1), zorder=9, linewidth=0.75)

    plt.legend()

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

