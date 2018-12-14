import nengo
import numpy as np

import nengo_solver_dales

from nengo.utils.least_squares_solvers import rmses

def test_accuracy():
    N = 100
    p_inh = 0.2
    sparsity = 0.5
    tols = [None, 1e-80, 1e-40, 1e-30, 1e-12,]# 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    model = nengo.Network(seed=0)
    with model:
        a = nengo.Ensemble(n_neurons=N, dimensions=3, seed=1)
        b = nengo.Ensemble(n_neurons=N, dimensions=3, seed=2)
        cs = [nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh, sparsity=sparsity, tol=tol)) for tol in tols]
    sim = nengo.Simulator(model)

    import pylab
    x, a = nengo.utils.ensemble.tuning_curves(a, sim, inputs=sim.data[a].eval_points)
    enc = sim.data[b].scaled_encoders
    target = np.dot(x, enc.T)

    ws = [sim.data[c].weights for c in cs]
    ts = [sim.data[c].solver_info['time'] for c in cs]

    actuals = [np.dot(a, w.T) for w in ws]

    rms = [np.mean(rmses(a, w.T, target)) for w in ws]

    print(rms)
    print(ts)

    for i in range(len(tols)):
        pylab.subplot(1, len(tols), i+1)
        pylab.scatter(target, actuals[i], s=1)
        pylab.title(rms[i])
    pylab.show()
