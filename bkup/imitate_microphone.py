#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Audio i/o, analysis, resynthesis. For [self.]

@author: Øyvind Brandtsegg, Axel Tidemann
@contact: obrandts@gmail.com, axel.tidemann@gmail.com
@license: GPL
'''

# Note: there are subtle errors in the Mac implementation of the
# multiprocessing module, this is due to the fact that the underlying
# system *requires* a fork() AND an exec() in order to function
# properly. If not, there will be problems related to certain external
# libraries. However, a workaround (that allows us to keep using the
# simple and nice multiprocessing module) is to import the troublesome
# libraries in their respective processes. This is why the code does not
# have all the imports at the beginning of the file. Under Windows, a
# separate process is spawned, so this is not an issue.

import multiprocessing as mp
from collections import deque

import numpy as np
from sklearn import preprocessing as pp

from communication import receive
from utils import net_rmse

def plot(learned_q, response_q):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.ion()
    plt.figure(figsize=(16,9)) # Change this if figure is too big (i.e. laptop)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,2])

    while True:
        plt.subplot(gs[0])
        plt.cla()
        plt.title('Learned.')
        plt.legend(plt.plot(learned_q.get()), ['level1', 'envelope1', 'pitch1', 'centroid1'])
        plt.draw()

        plt.subplot(gs[1])
        plt.cla()
        plt.title('Listening + response.')
        input_data, sound = response_q.get()
        plt.plot(np.vstack( [input_data, sound] ))
        ylim = plt.gca().get_ylim()
        plt.vlines(input_data.shape[0], ylim[0], ylim[1])
        plt.gca().annotate('response starts', xy=(input_data.shape[0],0), 
                           xytext=(input_data.shape[0] + 10, ylim[0] + .1))
        plt.gca().set_ylim(ylim)
        plt.draw()

        plt.tight_layout()

def learn(memorize_q, brain, learned_q):
    import Oger
    import mdp

    while True:
        input_data = memorize_q.get()
        scaler = pp.MinMaxScaler() 
        scaled_data = scaler.fit_transform(input_data)

        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                                  leak_rate=0.8, 
                                                  bias_scaling=.2, 
                                                  reset_states=False)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        net = mdp.hinet.FlowNode(reservoir + readout)

        x = scaled_data[:-1]
        y = scaled_data[1:]

        net.train(x,y)
        net.stop_training()
        
        brain.append((net, scaler))
        learned_q.put(scaled_data)

def respond(ear_q, brain, output, response_q):
    while True:
        signals = ear_q.get()
        rmse = net_rmse(brain, signals)
        print 'RMSE for neural networks in brain:', rmse
        net, scaler = brain[np.argmin(rmse)]
        
        input_data = scaler.transform(signals)
        sound = net(input_data)

        for row in scaler.inverse_transform(sound):
            output.append(row)

        response_q.put((input_data, sound))
        
def csound(memorize_q, ear_q, output, learn_state, respond_state):
    import csnd6
    cs = csnd6.Csound()
    cs.Compile("self_audio_csnd.csd")
    cs.Start()
    stopflag = 0
    #fft_audio_in1 = np.zeros(1024)
    #fft_audio_in2 = np.zeros(1024)
    offset = 0

    level1 = deque(maxlen=4000)
    envelope1 = deque(maxlen=4000)
    pitch1 = deque(maxlen=4000)
    centr1 = deque(maxlen=4000)

    while not stopflag:
        stopflag = cs.PerformKsmps()

        offset += 0.01
        offset %= 200
        cs.SetChannel("freq_offset", offset)
        #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
        #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

        # get Csound channel data
        #audioStatus = cs.GetChannel("audioStatus")
        level1.append(cs.GetChannel("level1"))
        envelope1.append(cs.GetChannel("envelope1"))
        pitch1.append(cs.GetChannel("pitch1"))
        centr1.append(cs.GetChannel("centroid1"))

        if learn_state.value:
            print '[self.] learns'
            memorize_q.put(np.asarray([ level1, envelope1, pitch1, centr1 ]).T)
            learn_state.value = 0
        if respond_state.value:
            print '[self.] responds'
            ear_q.put(np.asarray([ level1, envelope1, pitch1, centr1 ]).T)
            respond_state.value = 0

        try:
            response = output.pop(0)
            cs.SetChannel("respondLevel1", response[0])
            cs.SetChannel("respondEnvelope1", response[1])
            cs.SetChannel("respondPitch1", response[2])
            cs.SetChannel("respondCentroid1", response[3])
        except:
            cs.SetChannel("respondLevel1", 0)
            cs.SetChannel("respondEnvelope1", 0)
            cs.SetChannel("respondPitch1", 0)
            cs.SetChannel("respondCentroid1", 0)

class NeverUseMeAgainParser:
    def __init__(self, learn_state, respond_state):
        self.learn_state = learn_state
        self.respond_state = respond_state
        
    def parse(self, message):
        print '[self.] received:', message
        if message == 'learn':
            self.learn_state.value = 1
        if message == 'respond':
            self.respond_state.value = 1
            
if __name__ == '__main__':
    ear_q = mp.Queue()
    memorize_q = mp.Queue()
    learned_q = mp.Queue()
    response_q = mp.Queue()

    manager = mp.Manager()
    brain = manager.list()
    output = manager.list()

    learn_state = mp.Value('i', 0)
    respond_state = mp.Value('i', 0)
    parser = NeverUseMeAgainParser(learn_state, respond_state)
            
    mp.Process(target=learn, args=(memorize_q, brain, learned_q)).start()
    mp.Process(target=respond, args=(ear_q, brain, output, response_q)).start()
    mp.Process(target=plot, args=(learned_q, response_q)).start()
    mp.Process(target=csound, args=(memorize_q, ear_q, output, learn_state, respond_state)).start()
    mp.Process(target=receive, args=(parser.parse,)).start()

    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
