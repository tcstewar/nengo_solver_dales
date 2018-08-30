import nengo
import nengo_solver_dales
import numpy as np

def test_convert():
    model = nengo.Network()
    with model:
        stim = nengo.Node(lambda t: 0 if t<0.5 else 1)
        p_stim = nengo.Probe(stim)
        a = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(stim, a, synapse=None)
        
        bs = []
        p_bs = []
        conns = []
        for i in range(4):
            b = nengo.Ensemble(n_neurons=100, dimensions=1)
            bs.append(b)
            p_bs.append(nengo.Probe(b, synapse=0.02))
            conns.append(nengo.Connection(a, b,
                solver=nengo_solver_dales.DalesL2(),
                ))

        syn1 = 0.1
        syn2 = 0.01

        conns[0].synapse = syn1
        conns[1].synapse = syn2

    nengo_solver_dales.split_exc_inh(conns[2], model,
                                     exc_synapse=syn1, inh_synapse=syn2)
    nengo_solver_dales.split_exc_inh(conns[3], model,
                                     exc_synapse=syn2, inh_synapse=syn1)

    for conn in conns[2:]:
        assert conn not in model.all_connections

    sim = nengo.Simulator(model)
    sim.run(1.0)

    import pylab
    pylab.plot(sim.trange(), sim.data[p_stim])
    labels = ['%g'%syn1, '%g'%syn2, '%g+ %g-'% (syn1, syn2), '%g+ %g-' % (syn2, syn1)]
    for i, p_b in enumerate(p_bs):
        pylab.plot(sim.trange(), sim.data[p_b], label=labels[i])
    pylab.legend(loc='best')
    pylab.show()

