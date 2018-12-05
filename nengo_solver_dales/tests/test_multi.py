import nengo
import numpy as np

import nengo_solver_dales

def test_multi():
    p_inh = 0.2
    sparsity = 0.5
    model = nengo.Network(seed=0)
    with model:
        a = nengo.Ensemble(n_neurons=50, dimensions=3, seed=1)
        b = nengo.Ensemble(n_neurons=5, dimensions=3, seed=2)
        c1 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh, sparsity=sparsity, multiprocess=False), seed=1)
        c2 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh, sparsity=sparsity, multiprocess=True), seed=1)
    sim = nengo.Simulator(model)

    w1 = sim.data[c1].weights
    w2 = sim.data[c2].weights

    assert np.allclose(w1, w2)
