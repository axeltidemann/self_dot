#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
from uuid import uuid1
from collections import deque
import sys
import glob
import cPickle as pickle

import numpy as np
import zmq

from AI import learn, live
from utils import filesize, recv_array, send_array
from IO import MIC, CAMERA, STATE, SNAPSHOT, EVENT, EXTERNAL, send
            
def load_cns(prefix, brain_name):
    for filename in glob.glob(prefix+'*'):
        audio_recognizer, audio_producer, audio2video, scaler, host = pickle.load(file(filename, 'r'))
        name = filename[filename.rfind('.')+1:]
        mp.Process(target = live, 
                   args = (audio_recognizer, audio_producer, audio2video, scaler, host),
                   name = name).start()
        print 'Network loaded from file {} ({})'.format(filename, filesize(filename))
        send('decrement {}'.format(brain_name))

if __name__ == '__main__':
    host = sys.argv[1] if len(sys.argv) == 2 else 'localhost' 

    name = 'BRAIN'+str(uuid1())
    
    print '{} connecting to {}'.format(name, host)

    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, EXTERNAL))
    sender.send_json('register {} {}'.format(name, mp.cpu_count()))

    snapshot = context.socket(zmq.REQ)
    snapshot.connect('tcp://{}:{}'.format(host, SNAPSHOT))
    snapshot.send(b'Send me the state, please')
    state = snapshot.recv_json()

    poller = zmq.Poller()
    poller.register(mic, zmq.POLLIN)
    poller.register(camera, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    audio = deque()
    video = deque()

    audio_first_segment = []
    video_first_segment = []

    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()

        if mic in events:
            new_audio = recv_array(mic)
            if state['record']:
                audio.append(new_audio)
            
        if camera in events:
            new_video = recv_array(camera)
            if state['record']:
                video.append(new_video)
        
        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton and pushbutton['learn'] == name:
                mp.Process(target = learn, 
                           args = (audio_first_segment if len(audio_first_segment) else np.array(list(audio)), 
                                   np.array(list(audio)), 
                                   video_first_segment if len(video_first_segment) else np.array(list(video)),
                                   np.array(list(video)), 
                                   host),
                           name = 'NEURALNETWORK'+str(uuid1())).start()
                sender.send_json('decrement {}'.format(name))
                pushbutton['reset'] = True

            if 'reset' in pushbutton:
                audio.clear()
                video.clear()
                audio_first_segment = []
                video_first_segment = []
            
            if 'setmarker' in pushbutton:
                audio_first_segment = np.array(list(audio))
                video_first_segment = np.array(list(video))
                audio.clear()
                video.clear()

            if 'load' in pushbutton:
                load_cns(pushbutton['load'], name)
