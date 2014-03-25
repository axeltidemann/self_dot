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
    
def scale(dataset):
    results = []
    for data in dataset:
        scaler = preprocessing.MinMaxScaler()
        scaled_data = scaler.fit_transform(np.transpose(np.array([ data ] )))
        results.append( (scaled_data, scaler ))
    return results

def hear(ear_q, memorize_q):
    # This should be changed to use microphone input instead
    print 'I am listening'

    
def learn(memorize_q, brain_q, output):
    # Interestingly enough, importing Oger at the top of this file
    # causes troubles due to the parallelism.
    import Oger
    import mdp
    import matplotlib.pyplot as plt

    sound = []

    while True:
        sound = memorize_q.get()

        output.extend(range(10))
        for data, scaler in sound:
            plt.plot(data)
        plt.show()


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

    # while True:
        # set Csound channel data
        # cs.SetChannel("imitateLevel1", level1[-1])
        # cs.SetChannel("imitateEnvelope1", envelope1[-1])
        # cs.SetChannel("imitatePitch1", pitch1[-1])
        # cs.SetChannel("imitateCentroid1", centr1[-1])

        # input_photo = ear_q.get()
        # input_photo.shape = (1, input_photo.shape[0])

        # try:
        #     neural_net = brain_q.get_nowait()
        # except:
        #     pass
    
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
            print output.pop(0)
        except:
            continue
                
    
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), multiprocessing.active_children())


    

