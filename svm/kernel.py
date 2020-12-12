import numpy as np
from functools import wraps


"""Kernel Module

This Module implements kernel functions."""

def linear():
    """Linear kernel.

    Returns:
        closure: A Linear kernel K(x, y) = x.T * y
    """

    def linnear_kernel(x:np.ndarray, y:np.ndarray):

        return np.matmul(x, y.T)

    return linnear_kernel

def gaussian(sigma):
    """Gaussian kernel.

    Args:
        sigma: The standard error of the gaussian kernel.
        
    Returns:
        closure: A gaussian kernel K(x, y) = exp(- |x - y|^2 / 2 sigma^2)
    """

    g = -1.0 / (2 * (sigma ** 2))

    def gaussian_kernel(x, y):
        
        x_norm = np.sum(x**2, axis=1).reshape((-1, 1))
        y_norm = np.sum(y**2, axis=1).reshape((1, -1))
        w = -2 * np.matmul(x, y.T)
        w += x_norm
        w += y_norm
        
        return np.exp(w * g)

    return gaussian_kernel