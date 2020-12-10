import numpy as np


"""Solver Module

This Module implements SVM solving algorithms.
Currently only SMO is implemented."""


eps = 1e-15


class SVMSolver(object):
    
    def __init__(self):
        
        super(SVMSolver, self).__init__()

    
    def solve(self):
        
        raise NotImplementedError


class SMOSolver(SVMSolver):
    
    def __init__(self):
        super(SMOSolver, self).__init__()

    
    def solve(self, K:np.ndarray, y:np.ndarray, C:float):
        assert K.ndim == 2 and y.ndim == 1
        assert K.shape[0] == K.shape[1]
        assert K.shape[0] == y.shape[0]
        assert C > 0.0

        # initialization
        n = K.shape[0]
        self.K = K
        self.y = y
        self.C = C
        self.a = np.zeros_like(y, dtype=float)
        self.E = np.asarray(-self.y, dtype=float)
        self.b = 0.0
        
        unsolved = True
        while unsolved:
            
            unsolved = False

            # outer loop
            for i in range(n):
                
                if self.violated(i):

                    # inner loop
                    unsolved = True
                    j = np.argmax(np.abs(self.E-self.E[i]))
                    if isinstance(j, np.ndarray): j = j[np.random.randint(0, j.shape[0])]
                    j = int(j)

                    # solve 2 variable QP
                    self.solve2QP(i, j)
                             

        # build return value
        ind = np.arange(n)
        sv = ind[np.abs(self.a) > eps]
        ra = self.a[sv] * self.y[sv]

        return ra, self.b, sv

    
    def violated(self, index:int):
        a = self.a[index]
        det = self.y[index] * self.E[index]

        return (abs(a) < eps and det < -eps) or (abs(a-self.C) < eps and det > eps) \
                or (abs(det) > eps)

    
    def solve2QP(self, i:int, j:int):

        E1, E2 = self.E[i], self.E[j]
        a1, a2 = self.a[i], self.a[j]
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        y1, y2 = self.y[i], self.y[j]
        s = y1 * y2
        L = max(0.0, a1 - a2) if s < 0 else max(0.0, a1 + a2 - self.C)
        H = min(self.C, self.C + a1 - a2) if s < 0 else min(self.C, a1 + a2)
        
        # update a1
        if abs(eta) < eps:
            a1_new = L if E1 <= E2 else H
        else:
            a1_un = a1 + y1 * (E2 - E1) / eta
            if eta < -eps:
                a1_new = L if abs(L-a1_un) >= abs(H-a1_un) else H
            else:
                a1_new = a1_un
                if a1_new < L: a1_new = L
                if a1_new > H: a1_new = H
        
        # update a2
        da1 = a1_new - a1
        da2 = -s * da1
        a2_new = a2 + da2
        self.a[i] = a1_new
        self.a[j] = a2_new

        # update b
        db1 = -E1 - y1 * da1 * self.K[i, i] - y2 * da2 * self.K[i, j]
        if a1_new > eps and a1_new < self.C - eps:
            db = db1
        else:
            db2 = -E1 - y1 * da1 * self.K[i, j] - y2 * da2 * self.K[j, j]
            if a2_new > eps and a2_new < self.C - eps:
                db = db2
            else:
                db = (db1 + db2) / 2
        self.b = self.b + db
        
        # update E
        dE = y1 * da1 * self.K[i, : ] + y2 * da2 * self.K[j, : ] + db
        self.E = dE + self.E  
