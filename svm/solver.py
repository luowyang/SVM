"""Solver Module

This Module implements SVM solving algorithms.
Currently only SMO is implemented."""

import numpy as np

eps = 1e-3


class SVMSolver(object):
    
    def __init__(self):
        
        super(SVMSolver, self).__init__()

    
    def solve(self):
        
        raise NotImplementedError


class SMOSolver(SVMSolver):
    
    def __init__(self):
        super(SMOSolver, self).__init__()

    
    def solve(self, K:np.ndarray, y:np.ndarray, C:float, maxIter:int):
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
        self.E = -np.asarray(self.y, dtype=float)
        self.b = 0.0
        rng = np.arange(n)
        
        # outer loop
        iter = 0
        fullIter = True
        numChanged = 0
        while (iter < maxIter and numChanged > 0) or fullIter:
            numChanged = 0
            un1 = self.a > eps
            un2 = self.a < C - eps
            un = un1 * un2
            self.un = rng[un]

            if fullIter:
                for i in range(n):
                    numChanged += self.examineOuter(i)
            else:
                for i in self.un:
                    numChanged += self.examineOuter(i)
            
            if fullIter: fullIter = False
            elif numChanged == 0: fullIter = True

            iter += 1
            if iter % 50 == 0: print(iter)

        # build return value
        ind = np.arange(n)
        sv = ind[np.abs(self.a) > eps]
        ra = self.a[sv] * self.y[sv]

        return ra, self.b, sv

    
    def violated(self, idx:int):
        a = self.a[idx]
        det = self.y[idx] * self.E[idx]

        return (det < -eps and a < self.C) or (det > eps and a > 0)


    def examineOuter(self, idx:int):

        if self.violated(idx):

            # first choose j using heuristic
            if len(self.un) > 0:
                j = int(np.argmax(np.abs(self.E[self.un] - self.E[idx])))
                if self.solve2QP(idx, self.un[j]): return 1
            
            # if fails, loop j in unbounded set
            for j in self.un:
                if self.solve2QP(idx, j): return 1

            # if fails again, loop j in full set
            # if still fails, no change can be made
            for j in range(self.K.shape[0]):
                if self.solve2QP(idx, j): return 1
            
        # if no change is made, return 0
        return 0

    
    def solve2QP(self, i:int, j:int):

        # return 0 if i and j are the same
        if i == j: return 0

        E1, E2 = self.E[i], self.E[j]
        a1, a2 = self.a[i], self.a[j]
        y1, y2 = self.y[i], self.y[j]
        s = y1 * y2
        L = max(0.0, a1 - a2) if s < 0 else max(0.0, a1 + a2 - self.C)
        H = min(self.C, self.C + a1 - a2) if s < 0 else min(self.C, a1 + a2)
        # return 0 if no change
        if L == H: return 0
        
        # update a1
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
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
        da1 = a1_new - a1
        # return 0 if no change
        if abs(da1) < eps * (a1 + a1_new + eps): return 0
        self.a[i] = a1_new

        # update a2
        da2 = -s * da1
        a2_new = a2 + da2
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

        # print(da1)
        # print(da2)

        return 1