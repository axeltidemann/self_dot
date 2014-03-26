#!/usr/bin/python
# -*- coding: latin-1 -*-

''' 
Audio i/o, analysis, resynthesis. For [self.]

@author: Øyvind Brandtsegg, Axel Tidemann
@contact: obrandts@gmail.com, axel.tidemann@gmail.com
@license: GPL

Note: there are subtle errors in the Mac implementation of the multiprocessing
module, this is due to the fact that the underlying system *requires* a fork() AND 
an exec() in order to function properly. If not, there will be problems related to
external libraries (not the case for OpenCV, for instance, funny enough). However,
a strange workaround is to import the troublesome libraries in their
respective processes. This is why the code does not have all the imports at the
beginning of the file. CSound in particular, needs to be in the main thread. Otherwise,
epic fails ensue.
'''

import multiprocessing

import numpy as np
from sklearn import preprocessing
    
def hear(ear_q, memorize_q):
    # This should be changed to use microphone input instead
    print 'I am listening'

    
def learn(memorize_q, brain_q, output):
    import Oger
    import mdp
    import matplotlib.pyplot as plt

    while True:
        data, scaler = memorize_q.get()
        scaled = scaler.inverse_transform(data)

        for row in scaled:
            output.append(row)

        plt.ion()
        plt.figure('[self.] just learned these sequences. Scaled to unity.')
        plt.plot(data)
        plt.draw()
        

        # For illustration purposes, plot what self just learned.

        # reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, leak_rate=0.8, bias_scaling=.2) 
        # readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        # neural_net = mdp.hinet.FlowNode(reservoir + readout)

        # x = np.array(photos[:-1])
        # y = np.array(photos[1:])

        # neural_net.train(x,y)
        # neural_net.stop_training()

        # brain_q.put(neural_net)

        # print 'Learning phase completed, using', len(photos), 'photos.'

def play(ear_q, brain_q):
    neural_net = brain_q.get()

    # For illustration purposes: display what self heard, and what it predicted as graph plots 

    # while True:

        # input_photo = ear_q.get()
        # input_photo.shape = (1, input_photo.shape[0])

        # try:
        #     neural_net = brain_q.get_nowait()
        # except:
        #     pass

def scale(dataset):
    scaler = preprocessing.MinMaxScaler()
    all_data = np.hstack([ np.array( [data] ).T for data in dataset ])
    return scaler.fit_transform(all_data), scaler
    
if __name__ == '__main__':
    ear_q = multiprocessing.Queue()
    brain_q = multiprocessing.Queue()
    memorize_q = multiprocessing.Queue()

    manager = multiprocessing.Manager()
    output = manager.list()
            
    multiprocessing.Process(target=hear, args=(ear_q, memorize_q)).start()
    multiprocessing.Process(target=learn, args=(memorize_q, brain_q, output)).start()
    multiprocessing.Process(target=play, args=(ear_q, brain_q)).start()

    import csnd6
    cs = csnd6.Csound()
    cs.Compile("self_audio_csnd_test.csd")
    cs.Start()
    stopflag = 0
    #fft_audio_in1 = numpy.zeros(1024)
    #fft_audio_in2 = numpy.zeros(1024)
    offset = 0

    level1 = []
    envelope1 = []
    pitch1 = []
    centr1 = []

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
        # for now just copying the input values to test the signal chain

        i += 1
        if i == 1000:
            memorize_q.put(scale([ level1, envelope1, pitch1, centr1 ]))

        try:
            # set Csound channel data
            imitation = output.pop(0)
            print 'IMITATION', imitation
            cs.SetChannel("imitateLevel1", imitation[0])
            cs.SetChannel("imitateEnvelope1", imitation[1])
            cs.SetChannel("imitatePitch1", imitation[2])
            cs.SetChannel("imitateCentroid1", imitation[3])
        except:
            cs.SetChannel("imitateLevel1", 0)
            cs.SetChannel("imitateEnvelope1", 0)
            cs.SetChannel("imitatePitch1", 0)
            cs.SetChannel("imitateCentroid1", 0)
    
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), multiprocessing.active_children())


    

