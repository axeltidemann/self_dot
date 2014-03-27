#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Audio i/o, analysis, resynthesis. For [self.]

@author: Øyvind Brandtsegg, Axel Tidemann
@contact: obrandts@gmail.com, axel.tidemann@gmail.com
@license: GPL

Note: there are subtle errors in the Mac implementation of the
multiprocessing module, this is due to the fact that the underlying
system *requires* a fork() AND an exec() in order to function
properly. If not, there will be problems related to certain external
libraries. However, a workaround (that allows us to keep using the
simple and nice multiprocessing module) is to import the troublesome
libraries in their respective processes. This is why the code does not
have all the imports at the beginning of the file. CSound in
particular, needs to be in the main thread, and cannot be in another
process at all.

'''

import multiprocessing
from collections import deque

import numpy as np
from sklearn import preprocessing

def plot(learned_q, imitated_q):
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
        plt.title('Listening + imitation.')
        input_data, sound = imitated_q.get()
        plt.plot(np.vstack( [input_data, sound] ))
        ylim = plt.gca().get_ylim()
        plt.vlines(input_data.shape[0], ylim[0], ylim[1])
        plt.gca().annotate('imitation starts', xy=(input_data.shape[0],0), 
                           xytext=(input_data.shape[0] + 10, ylim[0] + .1))
        plt.gca().set_ylim(ylim)
        plt.draw()

        plt.tight_layout()

def learn(memorize_q, brain_q, learned_q):
    import Oger
    import mdp

    while True:
        input_data = memorize_q.get()
        scaler = preprocessing.MinMaxScaler() 
        scaled_data = scaler.fit_transform(input_data)

        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                                  leak_rate=0.8, 
                                                  bias_scaling=.2, 
                                                  reset_states=False)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        neural_net = mdp.hinet.FlowNode(reservoir + readout)

        x = scaled_data[:-1]
        y = scaled_data[1:]

        neural_net.train(x,y)
        neural_net.stop_training()
        
        brain_q.put((neural_net, scaler))
        learned_q.put(scaled_data)

def play(ear_q, brain_q, output, imitated_q):
    neural_net, scaler = brain_q.get()
    
    while True:
        input_data = scaler.transform(ear_q.get())
        sound = neural_net(input_data)

        for row in scaler.inverse_transform(sound):
            output.append(row)

        imitated_q.put((input_data, sound))

        try:
            neural_net, scaler = brain_q.get_nowait()
        except:
            pass
    
if __name__ == '__main__':
    ear_q = multiprocessing.Queue()
    brain_q = multiprocessing.Queue()
    memorize_q = multiprocessing.Queue()
    learned_q = multiprocessing.Queue()
    imitated_q = multiprocessing.Queue()
    
    output = multiprocessing.Manager().list()
            
    multiprocessing.Process(target=learn, args=(memorize_q, brain_q, learned_q)).start()
    multiprocessing.Process(target=play, args=(ear_q, brain_q, output, imitated_q)).start()
    multiprocessing.Process(target=plot, args=(learned_q, imitated_q)).start()

    import csnd6
    cs = csnd6.Csound()
    cs.Compile("self_audio_csnd.csd")
    cs.Start()
    stopflag = 0
    #fft_audio_in1 = np.zeros(1024)
    #fft_audio_in2 = np.zeros(1024)
    offset = 0

    level1 = deque(maxlen=1000)
    envelope1 = deque(maxlen=1000)
    pitch1 = deque(maxlen=1000)
    centr1 = deque(maxlen=1000)

    i = 0
    while not stopflag:
        stopflag = cs.PerformKsmps()

        offset += 0.01
        offset %= 200
        cs.SetChannel("freq_offset", offset)
        #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
        #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

        # get Csound channel data
        level1.append(cs.GetChannel("level1"))
        envelope1.append(cs.GetChannel("envelope1"))
        pitch1.append(cs.GetChannel("pitch1"))
        centr1.append(cs.GetChannel("centroid1"))

        # self does its imitation magic and writes new values to Csound channels

        i += 1
        if i == 1000: # These will be replaced with controls, i.e. someone saying LEARN! IMITATE!
            memorize_q.put(np.asarray([ level1, envelope1, pitch1, centr1 ]).T)

        if i == 2000:
            ear_q.put(np.asarray([ level1, envelope1, pitch1, centr1 ]).T)
            i = 0

        try:
            imitation = output.pop(0)
            cs.SetChannel("imitateLevel1", imitation[0])
            cs.SetChannel("imitateEnvelope1", imitation[1])
            cs.SetChannel("imitatePitch1", imitation[2])
            cs.SetChannel("imitateCentroid1", imitation[3])
        except:
            cs.SetChannel("imitateLevel1", 0)
            cs.SetChannel("imitateEnvelope1", 0)
            cs.SetChannel("imitatePitch1", 0)
            cs.SetChannel("imitateCentroid1", 0)

    map(lambda x: x.terminate(), multiprocessing.active_children())
