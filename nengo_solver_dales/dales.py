# based on https://github.com/nengo/nengo-extras/pull/70
#  which is based on https://github.com/nengo/nengo/issues/921


import time

from scipy.optimize import nnls
import numpy as np

import nengo
from nengo.params import NumberParam, BoolParam
from nengo.utils.least_squares_solvers import format_system, rmses


class DalesL2(nengo.solvers.Solver):
    """Solves for weights subject to Dale's principle."""

    p_inh = NumberParam('p_inh', low=0, high=1)   # proportion of inhibitory neurons
    reg = NumberParam('reg')                      # regularization
    keep_exc = BoolParam('keep_exc')              # whether to keep the excitatory connections
    keep_inh = BoolParam('keep_inh')              # whether to keep the inhibitory connections

    def __init__(self, p_inh=0.2, reg=0.1, keep_exc=True, keep_inh=True):
        super(DalesL2, self).__init__(weights=True)
        self.p_inh = p_inh
        self.reg = reg
        self.keep_exc = keep_exc
        self.keep_inh = keep_inh

    def __call__(self, A, Y, rng=None, E=None):
        tstart = time.time()
        Y, m, n, _, matrix_in = format_system(A, Y)
        
        sigma = A.max() * self.reg

        Y = self.mul_encoders(Y, E, copy=True)
        d = Y.shape[1]
        i = int(self.p_inh * n)
        
        A[:, :i] *= (-1)

        # form Gram matrix so we can add regularization
        GA = np.dot(A.T, A)
        np.fill_diagonal(GA, GA.diagonal() + A.shape[0] * sigma ** 2)
        GY = np.dot(A.T, Y)

        X = np.zeros((n, d))
        residuals = np.zeros(d)
        for j in range(d):
            X[:, j], residuals[j] = nnls(GA, GY[:, j])
        X[:i, :] *= (-1)
        A[:, :i] *= (-1)
        
        rms = rmses(A, X, Y)
        
        if not self.keep_inh:
            X[:i, :] = 0
        if not self.keep_exc:
            X[i:, :] = 0

        t = time.time() - tstart
        info = {'rmses': rms,
                'residuals': residuals,
                'time': t,
                'i': i}

        return X, info    
    
    
def convert_to_dales(conn, net, exc_synapse, inh_synapse, p_inh=0.2, reg=0.1):
    net.connections.remove(conn)
    with net:
        conn_exc = nengo.Connection(
            conn.pre, conn.post,
            solver=DalesL2(p_inh=p_inh, reg=reg, keep_exc=True, keep_inh=False),
            synapse=exc_synapse,
            function=conn.function,
            eval_points=conn.eval_points,
            scale_eval_points=conn.scale_eval_points,
            transform=conn.transform,
        )
        conn_inh = nengo.Connection(
            conn.pre, conn.post,
            solver=DalesL2(p_inh=p_inh, reg=reg, keep_exc=False, keep_inh=True),
            synapse=inh_synapse,
            function=conn.function,
            eval_points=conn.eval_points,
            scale_eval_points=conn.scale_eval_points,
            transform=conn.transform,
        )
    return conn_exc, conn_inh
