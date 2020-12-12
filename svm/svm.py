"""SVM Module

This module implements SVM class."""

import os
from types import FunctionType
import numpy as np
from math import inf

from .solver import SMOSolver


class SVM(object):
    """Support vector machine.

    A support vector machine, supporting soft margin and kernel method.
    """

    
    def __init__(self, kernel):

        assert isinstance(kernel, FunctionType)

        self.ker = kernel


    def save(self, path:str='.\\saved_model\\'):
        """Save model to file.

        Args:
            path: The path of the directory to save model to.
        """

        assert self.w is not None
        assert self.b is not None
        assert self.sv is not None

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'w.npy'), self.w)
        np.save(os.path.join(path, 'b.npy'), self.b)
        np.save(os.path.join(path, 'sv.npy'), self.sv)
        np.save(os.path.join(path, 'xs.npy'), self.xs)


    def load(self, path:str='.\\saved_model\\'):
        """Load model from file.

        Args:
            path: The path of the directory to load model from.
        """

        self.w = np.load(os.path.join(path, 'w.npy'))
        self.b = np.load(os.path.join(path, 'b.npy'))
        self.sv = np.load(os.path.join(path, 'sv.npy'))
        self.xs = np.load(os.path.join(path, 'xs.npy'))

    
    def predict(self, data:np.ndarray):
        """Predict the label of given data.

        Args:
            data: Some data to be predicted.

        Returns:
            int: The predicted label, +1 or -1.
        """
        
        assert self.w is not None
        assert self.b is not None
        assert self.sv is not None

        dot = self.ker(self.xs, data)
        wx = np.dot(self.w, dot)
        u = wx + self.b
        t = np.sign(u)
        
        return t

    
    def fit(self, data:np.ndarray, labels:np.ndarray, kmat:np.ndarray=None, \
            C:float=inf, solver:str="smo", maxIter:int=20000):
        """Train the SVM.

        Args:
            data: Training data.
            labels: Labels of corresponding training data.
            kmat: Kernel matrix of training data, kmat will not be recomputed when specified.
            C: Error parameter, ranged between 0 and infinity.
            solver: The solver used to train the SVM, default to use SMO solver.
            maxIter: Maximal number of iteraions of the solver.
        """
        
        assert data.shape[0] == labels.shape[0]

        self.K = kmat if kmat is not None else self.ker(data, data)
        self.x = data
        self.y = labels
        self.a = self.w = self.b = self.xs = None
        if C is None or C <= 0.0: C = inf

        print(self.K)

        if solver == "smo": solv = SMOSolver()
        else: raise Exception("Illegal argument: " + str)

        self.w, self.b, self.sv = solv.solve(self.K, self.y, C, maxIter)
        self.xs = self.x[self.sv]

        print("w:")
        print(self.w)
        print("b:")
        print(self.b)
        print("sv:")
        print(self.sv)
