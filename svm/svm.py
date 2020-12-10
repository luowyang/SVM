from types import FunctionType
import numpy as np
from math import inf

from .solver import SMOSolver

"""SVM Module

This module implements SVM class."""

class SVM(object):
    """Support vector machine.

    A support vector machine, supporting soft margin and kernel method.
    """

    
    def __init__(self, kernel):

        assert isinstance(kernel, FunctionType)

        self.ker = kernel

    
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
        wx = np.inner(self.w, dot)
        u = wx + self.b
        t = np.sign(u)
        
        return t

    
    def fit(self, data:np.ndarray, labels:np.ndarray, C:float, solver:str="smo"):
        """Train the SVM.

        Args:
            data: Training data.
            labels: Labels of corresponding training data.
            C: Error parameter, ranged between 0 and infinity.
            solver: The solver used to train the SVM, default to use SMO solver.
        """
        
        assert data.shape[0] == labels.shape[0]

        self.K = self.ker(data, data)
        self.x = data
        self.y = labels
        self.a = self.w = self.b = self.xs = None
        if C <= 0.0: C = inf

        print(self.K)

        if solver == "smo": solv = SMOSolver()
        else: raise Exception("Illegal argument: " + str)

        self.w, self.b, self.sv = solv.solve(self.K, self.y, C)
        self.xs = self.x[self.sv]

        print("w:")
        print(self.w)
        print("b:")
        print(self.b)
        print("sv:")
        print(self.sv)
