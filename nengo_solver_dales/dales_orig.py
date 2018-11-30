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

    # proportion of inhibitory neurons
    p_inh = NumberParam('p_inh', low=0, high=1)

    # amount of forced sparsity
    sparsity = NumberParam('sparsity', low=0, high=1)

    # regularization
    reg = NumberParam('reg')


    def __init__(self, p_inh=0.2, reg=0.1, sparsity=0.0):
        super(DalesL2, self).__init__(weights=True)
        self.p_inh = p_inh
        self.reg = reg
        self.sparsity = sparsity

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        
        sigma = A.max() * self.reg    # magnitude of noise

        Y = self.mul_encoders(Y, E, copy=True)
        n_post = Y.shape[1]
        n_inh = int(self.p_inh * n)

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma ** 2)
        GY = np.dot(A.T, Y)

        # flip the sign of the inhibitory neurons so we can do all
        #  the solving at once as a non-negative minimization
        GA[:, :n_inh] *= -1

        X = np.zeros((n, n_post))
        residuals = np.zeros(n_post)
        for j in range(n_post):
            if self.sparsity > 0:
                # choose random indices to keep
                N = GY.shape[0]
                S = N - int(N*self.sparsity)
                indices = rng.choice(np.arange(N), S, replace=False)
                sA = GA[indices, :][:, indices]
                sY = GY[indices, j]
            else:
                sA = GA
                sY = GY[:, j]
                indices = slice(None)

            # call nnls to do the non-negative least-squares minimization
            X[indices, j], residuals[j] = nnls(sA, sY)

        # flip the sign of the weights for the inhibitory neurons
        X[:n_inh, :] *= (-1)
        
        # compute the resulting rmse
        rms = rmses(A, X, Y)
        
        t = time.time() - tstart
        info = {'rmses': rms,
                'residuals': residuals/Y.shape[0],
                'time': t,
                'n_inh': n_inh}

        return X, info    


def split_exc_inh(conn, net, exc_synapse, inh_synapse):
    """Splits a nengo Connection into separate exc and inh Connections."""

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
