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
        
        return np.exp(np.abs(x - y) * g)

    return gaussian_kernel