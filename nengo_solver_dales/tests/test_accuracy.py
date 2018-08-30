import nengo
import numpy as np

import nengo_solver_dales

def test_accuracy():
    p_inh = 0.2
    model = nengo.Network(seed=0)
    with model:
        a = nengo.Ensemble(n_neurons=100, dimensions=3, seed=1)
        b = nengo.Ensemble(n_neurons=100, dimensions=3, seed=2)
        c3 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesL2(reg=0.1, p_inh=p_inh))
        c2 = nengo.Connection(a, b, solver=nengo_solver_dales.DalesNoise(reg=0.1, p_inh=p_inh))
        c1 = nengo.Connection(a, b, solver=nengo.solvers.LstsqL2(weights=True))
    sim = nengo.Simulator(model)

    rms1 = sim.data[c1].solver_info['rmses']
    rms2 = sim.data[c2].solver_info['rmses']
    rms3 = sim.data[c3].solver_info['rmses']

    print(sim.data[c2].solver_info['rmses'])
    print(sim.data[c2].solver_info['residuals'])
    print(sim.data[c2].solver_info['Y'])
    print(sim.data[c2].solver_info['Yhat'])
    print(sim.data[c3].solver_info['rmses'])
    print(sim.data[c3].solver_info['residuals'])
    print(sim.data[c3].solver_info['Y'])
    print(sim.data[c3].solver_info['Yhat'])

    print(np.mean(rms1), np.mean(rms2), np.mean(rms3))
    print(np.median(rms1), np.median(rms2), np.median(rms3))

    import pylab
    pylab.subplot(1,3,1)
    pylab.hist(rms1)
    pylab.subplot(1,3,2)
    pylab.hist(rms2)
    pylab.subplot(1,3,3)
    pylab.hist(rms3)
    pylab.show()
    1/0
    
        
