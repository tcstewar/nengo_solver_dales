import nengo
import numpy as np

import nengo_solver_dales

from nengo.utils.least_squares_solvers import rmses

def test_accuracy():
    p_inh = 0.2
    model = nengo.Network(seed=0)
    with model:
        a = nengo.Ensemble(n_neurons=100, dimensions=3, seed=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=3, seed=2)
        c1 = nengo.Connection(a, b, solver=nengo.solvers.LstsqL2(weights=True))
        c2 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh, sparsity=0.0))
        c3 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh, sparsity=0.2))
    sim = nengo.Simulator(model)

    import pylab
    x, a = nengo.utils.ensemble.tuning_curves(a, sim, inputs=sim.data[a].eval_points)
    enc = sim.data[b].scaled_encoders
    target = np.dot(x, enc.T)

    w1 = sim.data[c1].weights
    w2 = sim.data[c2].weights
    w3 = sim.data[c3].weights

    actual1 = np.dot(a, w1.T)
    actual2 = np.dot(a, w2.T)
    actual3 = np.dot(a, w3.T)

    rms1 = np.mean(rmses(a, w1.T, target))
    rms2 = np.mean(rmses(a, w2.T, target))
    rms3 = np.mean(rmses(a, w3.T, target))

    print(np.mean(rms1), np.mean(rms2), np.mean(rms3))

    assert rms1 < rms2 < rms3
    assert rms3 < rms1 * 5

    pylab.subplot(1, 3, 1)
    pylab.scatter(target, actual1, s=1)
    pylab.subplot(1, 3, 2)
    pylab.scatter(target, actual2, s=1)
    pylab.subplot(1, 3, 3)
    pylab.scatter(target, actual3, s=1)
    pylab.show()
