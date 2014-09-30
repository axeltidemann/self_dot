#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
from uuid import uuid1
from collections import deque
import sys
import glob
import cPickle as pickle
from subprocess import call
import time
import ctypes

import numpy as np
import zmq
from sklearn import preprocessing as pp
from sklearn import svm
from scipy.io import wavfile
from scikits.samplerate import resample
from scipy.signal import filtfilt

import utils
import IO
            
def train_network(x, y, output_dim=100, leak_rate=.9, bias_scaling=.2, reset_states=True, use_pinv=True):
    import Oger
    import mdp

    mdp.numx.random.seed(7)

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=output_dim, 
                                              leak_rate=leak_rate, 
                                              bias_scaling=bias_scaling, 
                                              reset_states=reset_states)
    readout = mdp.nodes.LinearRegressionNode(use_pinv=use_pinv)
        
    net = mdp.hinet.FlowNode(reservoir + readout)
    net.train(x,y)

    return net
        
def cochlear(filename, db=-40, stride=441, threshold=.025, new_rate=22050, ears=1, channels=71):
    rate, data = wavfile.read(filename)
    assert data.dtype == np.int16
    data = data / float(2**15)
    data = resample(utils.trim(data, threshold=threshold), float(new_rate)/rate, 'sinc_best')
    data = data*10**(db/20)
    utils.array_to_csv('{}-audio.txt'.format(filename), data)
    call(['./carfac-cmd', filename, str(len(data)), str(ears), str(channels), str(new_rate), str(stride)])
    naps = utils.csv_to_array('{}-output.txt'.format(filename))
    return np.sqrt(np.maximum(0, naps)/np.max(naps))

def classifier_brain(host):
    me = mp.current_process()
    me.name = 'BRAIN{}'.format(str(uuid1()))
    print '{} PID {} connecting to {}'.format(me.name, me.pid, host)

    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, IO.MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    speaker = context.socket(zmq.PUSH)
    speaker.connect('tcp://{}:{}'.format(host, IO.SPEAKER)) 

    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, IO.CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    projector = context.socket(zmq.PUSH)
    projector.connect('tcp://{}:{}'.format(host, IO.PROJECTOR)) 

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, IO.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))
    sender.send_json('register {} {}'.format(me.name, mp.cpu_count()))

    snapshot = context.socket(zmq.REQ)
    snapshot.connect('tcp://{}:{}'.format(host, IO.SNAPSHOT))
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
    wav_first_segment = []
    
    NAPs = []
    wavs = []

    audio_recognizer = []
    video_recognizer = []
    audio_producer = []
    video_producer = []

    maxlen = []
        
    import matplotlib.pyplot as plt
    plt.ion()

    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()

        if mic in events:
            new_audio = utils.recv_array(mic)
            if state['record']:
                audio.append(new_audio)
            
        if camera in events:
            new_video = utils.recv_array(camera)
            if state['record']:
                video.append(new_video)
        
        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton and pushbutton['learn'] == me.name:
                try:
                    filename = pushbutton['filename']
                    if len(wav_first_segment):
                        print 'Learning to associate {} -> {}'.format(wav_first_segment, filename)
                        filename = wav_first_segment
                    
                    utils.wait_for_wav(filename)
                    print 'Learning {} duration {} seconds'.format(filename, utils.wav_duration(filename))
                    start_time = time.time()
                    NAPs.append(cochlear(filename))
                    print 'Calculating cochlear neural activation patterns took {} seconds'.format(time.time() - start_time)

                    # plt.figure()
                    # plt.imshow(NAPs[-1].T, aspect='auto')
                    # plt.draw()

                    wavs.append(pushbutton['filename'] if len(wav_first_segment) else filename)

                    start_time = time.time()
                    maxlen = max([ memory.shape[0] for memory in NAPs ])
                    resampled_memories = [ resample(memory, float(maxlen)/memory.shape[0], 'sinc_best') for memory in NAPs ]

                    if len(np.unique([ m.shape[0] for m in resampled_memories ])) > 1:
                        # Actually, it might work better if they are all zero-padded instead of resampled. Or, build a classifier that deals with different lengths.
                        print 'Resampling yielded different lengths. Correcting by zero-padding the matrix.', [ m.shape[0] for m in resampled_memories ]
                        local_max = max([ m.shape[0] for m in resampled_memories ])
                        for i, m in enumerate(resampled_memories):
                            if m.shape[0] < local_max:
                                resampled_memories[i] = np.vstack(( m, np.zeros(( local_max - m.shape[0], m.shape[1])) ))

                    resampled_flattened_memories = [ np.ndarray.flatten(memory) for memory in resampled_memories ]

                    if len(NAPs) > 1:
                        audio_recognizer = svm.LinearSVC()
                        audio_recognizer.fit(resampled_flattened_memories, range(len(NAPs)))

                    video_segment = np.array(list(video))
                    NAP_len = NAPs[-1].shape[0]
                    video_len = video_segment.shape[0]
                    stride = int(max(1,np.floor(float(NAP_len)/video_len)))
                    x = NAPs[-1][:NAP_len - np.mod(NAP_len, stride*video_len):stride]
                    y = video_segment[:x.shape[0]]

                    tarantino = train_network(x,y)
                    tarantino.stride = stride
                    video_producer.append(tarantino)

                    print 'Learning classifier and video network in {} seconds'.format(time.time() - start_time)

                except Exception, e:
                    
                    print e, 'Learning aborted.'
                    valid_memories = min([ len(wavs), len(NAPs), len(video_producer) ])
                    wavs = wavs[:valid_memories]
                    NAPs = NAPs[:valid_memories]
                    video_producer = video_producer[:valid_memories]

                pushbutton['reset'] = True

            if 'respond' in pushbutton:
                print 'Respond to', pushbutton['filename']

                if len(NAPs) == 1:
                    sender.send_json('playfile {}'.format(wavs[-1]))
                    continue

                try:
                    utils.wait_for_wav(pushbutton['filename'])
                    NAP = cochlear(pushbutton['filename'])

                    # plt.figure()
                    # plt.imshow(NAP.T, aspect='auto')
                    # plt.draw()

                    NAP_resampled = resample(NAP, float(maxlen)/NAP.shape[0], 'sinc_best')

                    winner = audio_recognizer.predict(np.ndarray.flatten(NAP_resampled))[0]
                    
                    winnerinfo = open(wavs[winner][:-4]+'.txt', 'r')
                   
                    maxamp = 1
                    for line in winnerinfo:
                        if 'Max amp for file: ' in line:
                            maxamp = float(line[18:])
                    sender.send_json('playfile {} {}'.format(wavs[winner], maxamp))
                    
                    projection = video_producer[winner](NAP[::video_producer[winner].stride])

                    for row in projection:
                        utils.send_array(projector, row)

                except Exception, e:
                    print e, 'Response aborted.'

                pushbutton['reset'] = True


            if 'reset' in pushbutton:
                audio.clear()
                video.clear()
                audio_first_segment = []
                video_first_segment = []
                wav_first_segment = []
            
            if 'setmarker' in pushbutton:
                wav_first_segment = pushbutton['setmarker']
                audio_first_segment = np.array(list(audio))
                video_first_segment = np.array(list(video))
                audio.clear()
                video.clear()

            if 'load' in pushbutton:
                filename = pushbutton['load']
                audio_recognizer, audio_producer, video_producer, NAPs, wavs, maxlen = pickle.load(file(filename, 'r'))
                print 'Brain loaded from file {} ({})'.format(filename, utils.filesize(filename))

            if 'save' in pushbutton:
                filename = '{}.{}'.format(pushbutton['save'], me.name)
                pickle.dump((audio_recognizer, audio_producer, video_producer, NAPs, wavs, maxlen), file(filename, 'w'))
                print '{} saved as file {} ({})'.format(me.name, filename, utils.filesize(filename))
                
                        
if __name__ == '__main__':
    classifier_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
