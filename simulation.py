import brian2 as b2
b2.prefs.codegen.target = 'numpy'
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
import os.path
import random
import math


## Global Units ##
ms = b2.ms
mV = b2.mV
Hz = b2.Hz
amp = b2.amp
siemens = b2.siemens
nS = b2.nS
ohm = b2.ohm
second = b2.second
volt = b2.volt


# Global Parameters
El = -65 * mV
Vt = -55 * mV
taue = 2 * ms
taui = 25 * ms
taum = 20 * ms
E_exc = 0 * mV
E_inh = -80 * mV

# STDP params
tau_pre = 5 * ms
tau_post = 25 * ms
w_max = 0.5 * mV
A_pre = 0.01
A_post = -A_pre * tau_pre / tau_post * 1.05


def elapsed(sec):
    '''
    This function returns the elapsed time
    '''

    if sec < 60:
        return str(round(sec)) + ' secs'

    elif sec < 3600:
        return str(round((sec) / 60)) + ' mins'

    else:
        return str(round(sec / 3600)) + ' hrs'


def visualizeConnections(S):
    '''
     This function visualizes the connection
     between neurons
    '''

    src_len = len(S.source)
    tgt_len = len(S.target)

    print(src_len, tgt_len)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(np.zeros(src_len), np.arange(src_len), 'ok', ms=1)
    plt.plot(np.ones(tgt_len), np.arange(tgt_len), 'ok', ms=1)

    for i, j in zip(S.i, S.j):
            #print(i , j)
        plt.plot([0, 1], [i, j], '-k', ms=1)

    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(src_len, tgt_len))

    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok', ms=1)
    plt.xlim(-1, src_len)
    plt.ylim(-1, tgt_len)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')

    plt.show()


if __name__ == '__main__':

    start_time = time.time()

    circle_img = np.array([[0, 1, 1, 1, 0],
                          [1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [0, 1, 1, 1, 0]])


    # cross_img = np.array(cross_small)
    cross_img = np.array([[1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1]])

    # circle_img = np.where(circle_img > 1 , 0 , 1 )
    # cross_img = np.where(cross_img < 1 , 1 , 0)
    print(circle_img)

    print('Circle Dim: {}'.format(circle_img.shape))
    print('Cross Dim: {}'.format(cross_img.shape))

    # Plot the images
    fig1 = plt.figure(figsize=(10, 5))

    ax = fig1.add_subplot(121)
    plt.imshow(circle_img)
    plt.title("Circle")

    ax = fig1.add_subplot(122)
    plt.imshow(cross_img)
    plt.title("Cross")

    plt.show()

    # Number of neurons

    # Excitatory neurons
    e_neurons = 5

    # Ouput neurons
    o_neurons = 2

    num = e_neurons * e_neurons

    w_mm = 1/e_neurons

    # Excitatory synapse
    exc_lif_eqs = '''
    dv/dt = -((v - El) - (g_exc * (v - E_exc) / volt) - (g_inh * (v - E_inh) / volt))/taum : volt
    dg_exc/dt = -g_exc/taue: volt
    dg_inh/dt = -g_inh/taui: volt
    
    '''

    # Inhibitory synapse
    inh_lif_eqs = '''
    dv/dt = -((v - El) - (g_exc * (v - E_exc) / volt) - (g_inh * (v - E_inh) / volt))/taum : volt
    dg_inh/dt = -g_inh/taui: volt
    dg_exc/dt = -g_exc/taue: volt
    
    '''

    

    """
    out_lif_eqs = '''
    dv/dt = (g_exc + g_inh + El - v)/taum : volt
    dg_exc/dt = -g_exc/taue : volt
    dg_inh/dt = -g_inh/taui : volt
    '''
    """

    ##### GENERATE POISSON INPUT #####

    inp_neu = circle_img.shape[0] * circle_img.shape[1]
    inp_data = circle_img.reshape(inp_neu) * 255

    input_rate = inp_data * Hz + np.random.normal(0, 0.1, 25) * Hz


    # Input model
    inp = b2.PoissonGroup(num, rates = input_rate , name = 'inp')


    # Inhibitory group
    mem = b2.NeuronGroup(e_neurons ,
                         inh_lif_eqs ,
                         threshold = 'v > Vt' ,
                         reset = 'v = El' , 
                         refractory = 2 * ms ,
                         method = 'euler' ,
                         name = 'mem')


    mem.v = 'El + rand() * (v - El)'
    # mem.g_exc ='rand () * w_max'
    # mem.g_inh ='rand () * w_max'

    # Excitatory
    out = b2.NeuronGroup(o_neurons ,
                         exc_lif_eqs ,
                         threshold = 'v > Vt' ,
                         reset = 'v = El',
                         refractory = 2 * ms ,
                         method = 'euler' ,
                         name = 'out')


    out.v = 'El + rand() * (v - El)'
    # mem.g_inh = 'El + rand() * (v - El)'
    # out.g_inh = 'El + rand() * (v - El)'
    


    # Input to Memory
    Syn_inp_mem = b2.Synapses(inp ,
                              mem ,
                              model =
                              ''' w: volt
                                  da_src/dt = -a_src/tau_pre : volt (clock-driven)
                                  da_tgt/dt = -a_tgt/tau_post : volt (clock-driven)
                              ''' ,
                              on_pre = '''
                              g_exc += w
                              a_src += A_pre * mV
                              w = clip(w + a_tgt , 0 * mV , w_max)
                              ''' ,

                              on_post = '''
                              a_tgt += A_post * mV
                              w = clip(w + a_src , 0 * mV  , w_max)
                              ''' ,
                              method = 'euler' , 

                              name = 'Syn_inp_mem'

                              )

    # Input to Output
    Syn_inp_out = b2.Synapses(inp , out , 'w : volt' , on_pre = 'g_exc += w')

    # Input to Input
    Syn_inp_inp = b2.Synapses(inp , inp)

    # Memory to Memory
    Syn_mem_mem = b2.Synapses(mem , mem , 'w: volt' , on_pre = 'g_exc += w')

    # Memory to Output
    Syn_mem_out = b2.Synapses(mem , out , 'w: volt' , on_pre = 'g_exc += w')

    # Output to Output
    Syn_out_out = b2.Synapses(out , out , 'w: volt' , on_pre = 'g_inh -= w')

    
    # Set up connections
    Syn_inp_inp.connect()
    Syn_inp_mem.connect()
    Syn_inp_out.connect()
    Syn_mem_mem.connect()
    Syn_mem_out.connect()
    Syn_out_out.connect()

    Syn_inp_mem.w = 'rand() * w_max'

    # Set up monitors
    # spike_mon_poi = b2.SpikeMonitor(poi)
    spike_mon_inp = b2.SpikeMonitor(inp)
    spike_mon_mem = b2.SpikeMonitor(mem)
    spike_mon_out = b2.SpikeMonitor(out)
    
    
    # Define State monitor
    state_mon_inp = b2.StateMonitor(Syn_mem_mem, ['g_exc'], record = True, name='State_Mon_inp')
    state_mon_inh = b2.StateMonitor(Syn_out_out, ['g_inh'], record = True, name='State_Mon_inh')
    state_mon_mem = b2.StateMonitor(Syn_inp_mem, ['w'], record = Syn_inp_mem['w > 0 * mV'], name='State_mon_mem')
        


    # Create a network
    net = b2.Network([inp , #spike_neurons_inp,
                  mem,
                  out,
                  Syn_inp_inp,
                  Syn_inp_mem,
                  Syn_inp_out,
                  Syn_mem_mem,
                  Syn_mem_out,
                  Syn_out_out,
                  #spike_mon_poi,
                  spike_mon_inp,
                  spike_mon_mem,
                  spike_mon_out,
                  state_mon_inp,
                  state_mon_mem])


    net.store()
    
    total_time = 3000 * ms 
    # neu.rate = inp_data  * Hz


    for x in range(3):
        net.run((total_time) , report = 'text')

    print('Output Spike train: ' , spike_mon_out.spike_trains() , spike_mon_out.all_values())
    print('Output Spikes: ' , spike_mon_out.num_spikes)
    print('Inh Spikes: ' , spike_mon_mem.num_spikes)

    print(mem.g_inh)
    print(mem.g_exc)
    print(out.g_inh)


    fig = plt.figure(figsize=(20,5))

    ax = fig.add_subplot(311)
    plt.plot(spike_mon_mem.t/ms, spike_mon_mem.i, '|' , linewidth = 0.5)
    plt.title('Inh Spikes')
    plt.tight_layout()


    ax = fig.add_subplot(312)
    plt.plot(spike_mon_inp.t/ms, spike_mon_inp.i, '|' , linewidth = 0.1)
    plt.title('Inp Spikes')
    plt.tight_layout()

    ax = fig.add_subplot(313)
    plt.plot(state_mon_inp.t / second, state_mon_inp.g_exc[0] , label = 'Exc' , color = 'orange' , linewidth = 0.2)
    plt.plot(state_mon_inh.t / second, state_mon_inh.g_inh[0] , label = 'Inh' , color = 'blue')
    plt.legend(loc = 'best')
    ax.set_xlabel('Exc vs Inh current')
    plt.tight_layout()


    plt.show()