# based on https://github.com/nengo/nengo-extras/pull/70
#  which is based on https://github.com/nengo/nengo/issues/921

import time

from scipy.optimize import nnls
import numpy as np

import nengo
from nengo.params import NumberParam, BoolParam
from nengo.solvers import SolverParam
from nengo.utils.least_squares_solvers import format_system, rmses

class SolverSet(nengo.solvers.Solver):
    solver = SolverParam('solver')

    def __init__(self, solver, limit=None):
        super(SolverSet, self).__init__(weights=solver.weights)
        self.solver = solver
        self.limit = limit
        self.count = 0
        self.computed = False

    def __call__(self, A, Y, rng=None, E=None):
        self.count += 1
        if self.count > self.limit:
            raise Exception('Too many Connections using the same SolverSet')
        if not self.computed:
            self.result = self.solver(A, Y, rng=rng, E=E)
        return self.result

class PositiveOnly(nengo.solvers.Solver):
    solver_set = SolverParam('solver_set')

    def __init__(self, solver_set):
        super(PositiveOnly, self).__init__(weights=solver_set.weights)
        self.solver_set = solver_set

    def __call__(self, A, Y, rng=None, E=None):
        dec, info = self.solver_set(A, Y, rng=rng, E=E)
        dec = np.maximum(dec, 0)
        return dec, info

class NegativeOnly(nengo.solvers.Solver):
    solver_set = SolverParam('solver_set')

    def __init__(self, solver_set):
        super(NegativeOnly, self).__init__(weights=solver_set.weights)
        self.solver_set = solver_set

    def __call__(self, A, Y, rng=None, E=None):
        dec, info = self.solver_set(A, Y, rng=rng, E=E)
        dec = np.minimum(dec, 0)
        return dec, info


class DalesL2(nengo.solvers.Solver):
    """Solves for weights subject to Dale's principle."""

    p_inh = NumberParam('p_inh', low=0, high=1)   # proportion of inhibitory neurons
    reg = NumberParam('reg')                      # regularization

    def __init__(self, p_inh=0.2, reg=0.1):
        super(DalesL2, self).__init__(weights=True)
        self.p_inh = p_inh
        self.reg = reg

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        
        sigma = A.max() * self.reg

        Y = self.mul_encoders(Y, E, copy=True)
        d = Y.shape[1]
        n_inh = int(self.p_inh * n)
        

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma ** 2)
        GY = np.dot(A.T, Y)

        GA[:, :n_inh] *= (-1)

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for j in range(d):
            X[:, j], residuals[j] = nnls(GA, GY[:, j])


        X[:n_inh, :] *= (-1)
        
        rms = rmses(A, X, Y)
        
        t = time.time() - tstart
        info = {'rmses': rms,
                'residuals': residuals/Y.shape[0],
                'Y': Y,
                'Yhat': np.dot(A, X),
                'time': t,
                'n_inh': n_inh}

        return X, info    


class DalesNoise(nengo.solvers.Solver):
    p_inh = NumberParam('p_inh', low=0, high=1)   # proportion of inhibitory neurons
    reg = NumberParam('reg')                      # regularization

    def __init__(self, p_inh=0.2, reg=0.1):
        super(DalesNoise, self).__init__(weights=True)
        self.p_inh = p_inh
        self.reg = reg

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
                                
        sigma = self.reg * A.max()
        A_noise = A + rng.normal(scale=sigma, size=A.shape)        

        Y = self.mul_encoders(Y, E, copy=True)
        d = Y.shape[1]
        n_inh = int(self.p_inh * n)

        A_noise[:, :n_inh] *= (-1)
        X = np.zeros((n, d))

        residuals = np.zeros(d)
        for j in range(d):
            X[:, j], residuals[j] = nnls(A_noise, Y[:, j])

        X[:n_inh, :] *= (-1)

        t = time.time() - tstart
        info = {'rmses': rmses(A, X, Y),
                'residuals': residuals/Y.shape[0],
                'Y': Y,
                'Yhat': np.dot(A, X),
                'time': t,
                'n_inh': n_inh}

        return X, info
    
def split_exc_inh(conn, net, exc_synapse, inh_synapse):
    net.connections.remove(conn)

    solver_set = SolverSet(conn.solver, limit=2)

    with net:
        conn_exc = nengo.Connection(
            conn.pre, conn.post,
            solver=PositiveOnly(solver_set),
            synapse=exc_synapse,
            function=conn.function,
            eval_points=conn.eval_points,
            scale_eval_points=conn.scale_eval_points,
            transform=conn.transform,
        )
        conn_inh = nengo.Connection(
            conn.pre, conn.post,
            solver=NegativeOnly(solver_set),
            synapse=inh_synapse,
            function=conn.function,
            eval_points=conn.eval_points,
            scale_eval_points=conn.scale_eval_points,
            transform=conn.transform,
        )
    return conn_exc, conn_inh
