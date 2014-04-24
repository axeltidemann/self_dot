#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import multiprocessing as mp
from collections import deque

import numpy as np
from sklearn import preprocessing as pp
import cv2

from communication import receive as receive_messages
from utils import net_rmse, Parser

# The new maxlen parameter, since deque cannot be shared, and we use a
# list with shared memory instead.
MAXLEN_AUDIO = 4000
MAXLEN_VIDEO = 100

def video(camera, projector):
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    video_feed = cv2.VideoCapture(0)

    while True:
        _, frame = video_feed.read()
        frame = cv2.resize(frame, (320,180))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera.append(np.ndarray.flatten(gray_image)/255.)

        if len(camera) > MAXLEN_VIDEO: 
            del camera[0]
            
        try:
            cv2.imshow('Output', np.reshape(projector.pop(0), (180,320)))
        except:
            cv2.imshow('Output', np.zeros((180,320)))

        cv2.waitKey(100)

def audio(mic, speaker):
    import csnd6
    cs = csnd6.Csound()
    cs.Compile("self_audio_csnd.csd")
    cs.Start()
    stopflag = 0
    #fft_audio_in1 = np.zeros(1024)
    #fft_audio_in2 = np.zeros(1024)
    offset = 0

    while not stopflag:
        stopflag = cs.PerformKsmps()

        offset += 0.01
        offset %= 200
        cs.SetChannel("freq_offset", offset)
        #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
        #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

        # get Csound channel data
        #audioStatus = cs.GetChannel("audioStatus")
        mic.append([ cs.GetChannel("level1"), cs.GetChannel("envelope1"),
                     cs.GetChannel("pitch1"), cs.GetChannel("centroid1") ])

        if len(mic) > MAXLEN_AUDIO:
            del mic[0]
            
        try:
            response = speaker.pop(0)
            cs.SetChannel("respondLevel1", response[0])
            cs.SetChannel("respondEnvelope1", response[1])
            cs.SetChannel("respondPitch1", response[2])
            cs.SetChannel("respondCentroid1", response[3])
        except:
            cs.SetChannel("respondLevel1", 0)
            cs.SetChannel("respondEnvelope1", 0)
            cs.SetChannel("respondPitch1", 0)
            cs.SetChannel("respondCentroid1", 0)

def learn(state_q, mic, camera, brain):
    import Oger
    import mdp

    while True:
        state_q.get()

        audio_data = np.asarray(mic)
        audio_data = np.resize(audio_data, (MAXLEN_AUDIO, len(mic[0])))
        video_data = np.asarray(camera)
        video_data = np.resize(video_data, (MAXLEN_VIDEO, len(camera[0])))
                
        scaler = pp.MinMaxScaler() 
        scaled_data = scaler.fit_transform(audio_data)

        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                                  leak_rate=0.8, 
                                                  bias_scaling=.2, 
                                                  reset_states=True)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        audio_net = mdp.hinet.FlowNode(reservoir + readout)

        x = scaled_data[:-1]
        y = scaled_data[1:]

        audio_net.train(x,y)
        audio_net.stop_training()

        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=500, 
                                                  leak_rate=0.8, 
                                                  bias_scaling=.2, 
                                                  reset_states=True)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        video_net = mdp.hinet.FlowNode(reservoir + readout)

        # Audio is associated with video, but due to the higher
        # sampling frequency of the sound compared to video, we make a
        # selection of the audio data. 
        stride = audio_data.shape[0]/video_data.shape[0]
        x = scaled_data[::stride]
        y = video_data

        video_net.train(x,y)
        video_net.stop_training()
        
        brain.append((audio_net, video_net, scaler))

        print 'Finished learning audio-video association'

def respond(state_q, mic, speaker, camera, projector, brain):
    while True:
        state_q.get()
        audio_data = np.asarray(mic)
        audio_data = np.resize(audio_data, (MAXLEN_AUDIO, len(mic[0])))
        video_data = np.asarray(camera)
        video_data = np.resize(video_data, (MAXLEN_VIDEO, len(camera[0])))

        rmse = net_rmse([ (net, scaler) for net,_,scaler in brain ], audio_data)
        print 'RMSE for neural networks in brain:', rmse
        audio_net, video_net, scaler = brain[np.argmin(rmse)]
        
        input_data = scaler.transform(audio_data)
        sound = audio_net(input_data)

        for row in scaler.inverse_transform(sound):
            speaker.append(row)

        stride = audio_data.shape[0]/video_data.shape[0]
        projection = video_net(input_data[::stride]) 

        for row in projection:
            projector.append(row)

if __name__ == '__main__':
    learn_state = mp.Queue()
    respond_state = mp.Queue()

    parser = Parser(learn_state, respond_state)

    manager = mp.Manager()

    brain = manager.list()
    mic = manager.list()
    speaker = manager.list()
    camera = manager.list()
    projector = manager.list()
    
    mp.Process(target=audio, args=(mic, speaker)).start()
    mp.Process(target=video, args=(camera, projector)).start()
    mp.Process(target=learn, args=(learn_state, mic, camera, brain)).start()
    mp.Process(target=respond, args=(respond_state, mic, speaker, camera, projector, brain)).start()
    mp.Process(target=receive_messages, args=(parser.parse,)).start()
    
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())

