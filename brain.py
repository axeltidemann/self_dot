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

import numpy as np
import zmq
from sklearn import preprocessing as pp
from sklearn import svm
from scipy.io import wavfile
from scikits.samplerate import resample
from scipy.signal import filtfilt

from AI import learn, live, recognize, _train_network
from utils import filesize, recv_array, send_array, trim, array_to_csv, csv_to_array, wait_for_wav
from IO import MIC, CAMERA, STATE, SNAPSHOT, EVENT, EXTERNAL, send, SPEAKER, PROJECTOR
            
def load_cns(prefix, brain_name):
    for filename in glob.glob(prefix+'*'):
        audio_recognizer, audio_producer, audio2video, scaler, host = pickle.load(file(filename, 'r'))
        name = filename[filename.rfind('.')+1:]
        mp.Process(target = AI.live, 
                   args = (audio_recognizer, audio_producer, audio2video, scaler, host),
                   name = name).start()
        print 'Network loaded from file {} ({})'.format(filename, filesize(filename))
        send('decrement {}'.format(brain_name))

        
def cochlear(filename, db = -40, stride = 441, new_rate = 22050, threshold=.025):
    rate, data = wavfile.read(filename)
    assert data.dtype == np.int16
    data = data / float(2**15)
    data = resample(trim(data, threshold=threshold), float(new_rate)/rate, 'sinc_best')
    data = data*10**(db/20)
    array_to_csv('{}-audio.txt'.format(filename), data)
    call(['./carfac-cmd', filename, str(len(data))])
    carfac = csv_to_array('{}-output.txt'.format(filename))
    smooth = filtfilt([1], [1, -.995], carfac, axis=0)
    decim = smooth[::stride]
    return np.sqrt(np.maximum(0, decim)/np.max(decim))


def classifier_brain(host):
    me = mp.current_process()
    me.name = 'BRAIN{}'.format(str(uuid1()))
    print '{} PID {} connecting to {}'.format(me.name, me.pid, host)

    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    speaker = context.socket(zmq.PUSH)
    speaker.connect('tcp://{}:{}'.format(host, SPEAKER)) 

    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    projector = context.socket(zmq.PUSH)
    projector.connect('tcp://{}:{}'.format(host, PROJECTOR)) 

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, EXTERNAL))
    sender.send_json('register {} {}'.format(me.name, mp.cpu_count()))

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
    
    audio_memories = []
    video_memories = []
    wav_memories = []

    audio_recognizer = []
    audio_producer = []
    video_producer = []

    idxs = [0,6,7,8,9,12]
    maxlen = []
        
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
            if 'learn' in pushbutton and pushbutton['learn'] == me.name:

                print 'Learning', pushbutton['wavfile']
                wait_for_wav(pushbutton['wavfile'])

                start_time = time.time()
                try:
                    audio_memories.append(cochlear(pushbutton['wavfile']))
                except Exception, e:
                    print e
                    print 'Learning aborted.'
                    continue
                cochlear_calculation_time = time.time() - start_time

                wav_memories.append(pushbutton['wavfile'])

                maxlen = max([ memory.shape[0] for memory in audio_memories ])
                resampled_memories = [ resample(memory, float(maxlen)/memory.shape[0], 'sinc_best') for memory in audio_memories ]
                resampled_flattened_memories = [ np.ndarray.flatten(memory) for memory in resampled_memories ]

                if len(audio_memories) > 1:
                    audio_recognizer = svm.LinearSVC()
                    audio_recognizer.fit(resampled_flattened_memories, range(len(audio_memories)))

                print 'Calculating cochlear neural activation patterns took {} seconds'.format(cochlear_calculation_time)
                # start_time = time.time()
                # audio_segment = np.array(list(audio))
                # video_segment = np.array(list(video))
                # scaler = pp.MinMaxScaler()
                # scaled_audio = scaler.fit_transform(audio_segment)

                # # plt.figure()
                # # plt.plot(chop(scaled_audio[:,idxs]))
                # # plt.draw()

                # audio_producer.append(_train_network([scaled_audio[:-1]], [scaled_audio[1:]]))

                # audio_memories.append(scaled_audio[:,idxs])#[:len(audio)/2,idxs])
                # video_memories.append(video_segment)

                # targets = range(len(audio_memories))

                # maxlength = max(map(lambda x: x.shape[0], audio_memories))

                # train_data = [ np.ndarray.flatten(np.vstack((memory, np.zeros((maxlength - memory.shape[0], memory.shape[1]))))) for memory in audio_memories ]                
                                
                # audio_recognizer = GaussianNB()
                # audio_recognizer.fit(train_data, targets)
                
                # print 'Testing recently learned audio segments: {}% correct'\
                #     .format(np.mean([ i == audio_recognizer.predict(np.ndarray.flatten(np.vstack((memory, np.zeros((maxlength - memory.shape[0], memory.shape[1]))))))[0] for i, memory in enumerate(audio_memories) ])*100)

                # stride = scaled_audio.shape[0]/video_segment.shape[0]
                
                # x = [ scaled_audio[scaled_audio.shape[0] - stride*video_segment.shape[0]::stride] ]
                # y = [ video_segment ]
                # video_producer.append(_train_network(x,y, output_dim=100))
                
                # print 'Lessons learned in {} seconds'.format(time.time() - start_time)

                pushbutton['reset'] = True

            if 'rmse' in pushbutton and len(audio):
                print 'RESPOND to', pushbutton['wavfile']

                if len(audio_memories) == 1:
                    sender.send_json('playfile {}'.format(wav_memories[-1]))
                else:
                    wait_for_wav(pushbutton['wavfile'])
                    test = cochlear(pushbutton['wavfile'])
                    test = resample(test, float(maxlen)/test.shape[0], 'sinc_best')
                    winner = audio_recognizer.predict(np.ndarray.flatten(test))[0]
                    sender.send_json('playfile {}'.format(wav_memories[winner]))

                # video_segment = np.array(list(video))
                # audio_segment = np.array(list(audio))
                # scaler = pp.MinMaxScaler()
                # scaled_audio = scaler.fit_transform(audio_segment)

                # # plt.figure()
                # # plt.plot(scaled_audio[:minlength,idxs])
                # # plt.draw()
                
                # selectah = np.ndarray.flatten( scaled_audio[:maxlength, idxs] if scaled_audio.shape[0] > maxlength else np.vstack( (scaled_audio[:,idxs], np.zeros(( maxlength - scaled_audio.shape[0], len(idxs) )))) )

                # winner = audio_recognizer.predict(selectah)[0]
                # print 'WINNER NETWORK', winner

                # sound = audio_producer[winner](scaled_audio)

                # stride = audio_segment.shape[0]/video_segment.shape[0]

                # projection = video_producer[winner](audio_segment[audio_segment.shape[0] - stride*video_segment.shape[0]::stride])

                # for row in projection:
                #     send_array(projector, row)

                # for row in scaler.inverse_transform(sound):
                #     send_array(speaker, row)

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
                load_cns(pushbutton['load'], me.name)
                
                        
if __name__ == '__main__':
    #start_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    #monolithic_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    classifier_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
