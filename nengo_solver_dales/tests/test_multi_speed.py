import nengo
import numpy as np
import timeit

import nengo_solver_dales

def test_multi_speed():
    p_inh = 0.2
    sparsity = 0.5

    Ns = [200, 300, 400, 500]
    ts = {}
    ts[True] = []
    ts[False] = []

    for N in Ns:
        w = None
        for multi in [True, False]:
            model = nengo.Network(seed=0)
            with model:
                a = nengo.Ensemble(n_neurons=N, dimensions=3)
                b = nengo.Ensemble(n_neurons=N, dimensions=3)
                c = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh,
                                                                         sparsity=sparsity, multiprocess=multi))
            t_start = timeit.default_timer()
            sim = nengo.Simulator(model)
            t = timeit.default_timer() - t_start
            ts[multi].append(t)

            if w is None:
                w = sim.data[c].weights
            else:
                assert np.allclose(w, sim.data[c].weights)

    import pylab
    pylab.loglog(Ns, ts[False], label='serial')
    pylab.loglog(Ns, ts[True], label='multi')
    pylab.xlabel('n_neurons')
    pylab.ylabel('time')
    pylab.legend(loc='best')
    pylab.show()
