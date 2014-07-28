#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
from uuid import uuid1
from collections import deque
import sys
import glob
import cPickle as pickle
import time

import numpy as np
import zmq
from sklearn import preprocessing as pp
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

from AI import learn, live, recognize, _train_network
from utils import filesize, recv_array, send_array
from IO import MIC, CAMERA, STATE, SNAPSHOT, EVENT, EXTERNAL, send, RECOGNIZE_IN, RECOGNIZE_LEARN, SPEAKER, PROJECTOR
            
def load_cns(prefix, brain_name):
    for filename in glob.glob(prefix+'*'):
        audio_recognizer, audio_producer, audio2video, scaler, host = pickle.load(file(filename, 'r'))
        name = filename[filename.rfind('.')+1:]
        mp.Process(target = AI.live, 
                   args = (audio_recognizer, audio_producer, audio2video, scaler, host),
                   name = name).start()
        print 'Network loaded from file {} ({})'.format(filename, filesize(filename))
        send('decrement {}'.format(brain_name))

        
def start_brain(host):
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

    rec_in = context.socket(zmq.PUB)
    rec_in.bind('tcp://*:{}'.format(RECOGNIZE_IN))

    rec_learn = context.socket(zmq.PUB)
    rec_learn.bind('tcp://*:{}'.format(RECOGNIZE_LEARN))
        
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

    # mp.Process(target = recognize,
    #            args = ('localhost',),
    #            name = 'RECOGNIZER').start()
    
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

                #send_array(rec_learn, audio_first_segment if len(audio_first_segment) else np.array(list(audio)))
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

            if 'findwinner' in pushbutton:
                if len(audio):
                    send_array(rec_in, audio_first_segment if len(audio_first_segment) else np.array(list(audio)))

def monolithic_brain(host):
    name = 'BRAIN'+str(uuid1())
    
    print '{} connecting to {}'.format(name, host)

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
    
    audio_memories = []
    video_memories = []

    audio_recognizer = []
    audio_producer = []
    video_producer = []

    import matplotlib.pyplot as plt
    plt.ion()

    legends= ["level1", 
                                        "pitch1ptrack", 
                                        "pitch1pll", 
                                        "autocorr1", 
                                        "centroid1",
                                        "spread1", 
                                        "skewness1", 
                                        "kurtosis1", 
                                        "flatness1", 
                                        "crest1", 
                                        "flux1", 
                                        "epochSig1", 
                                        "epochRms1", 
                                        "epochZCcps1"]

    idxs = [0,6,7,8,9,12]
    #idxs = range(14) # Uncomment to include all parameters
    
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

                start_time = time.time()
                audio_segment = np.array(list(audio))
                video_segment = np.array(list(video))
                scaler = pp.MinMaxScaler()
                scaled_audio = scaler.fit_transform(audio_segment)

                # plt.figure()
                # for i, l in zip(idxs, legends):
                #     plt.plot(scaled_audio[:,i], label=l)
                # plt.legend()
                # plt.draw()

                audio_producer.append(_train_network([scaled_audio[:-1]], [scaled_audio[1:]]))

                audio_memories.append(chop(scaled_audio[:, idxs ]))#[:len(audio)/2,idxs])
                video_memories.append(video_segment)

                plt.figure()
                plt.plot(audio_memories[-1])
                plt.draw()

                targets = []
                for i, memory in enumerate(audio_memories):
                    target = np.zeros((memory.shape[0], len(audio_memories))) - 1
                    target[:,i] = 1
                    targets.append(target)

                #the_message = zip(audio_memories, targets)
                #the_message *= 10
                #np.random.shuffle(the_message)
                #inputs, targets = map(np.vstack, zip(*the_message))
                #inputs += .3*np.random.random_sample(inputs.shape) - .3 # 20% noise

                # Parameters to network found by evolution: 58, 12, 98
                audio_recognizer = _train_network(audio_memories, targets, output_dim=200, leak_rate=.12, bias_scaling=.98) 

                print 'Testing recently learned audio segments: {}% correct'\
                    .format(np.mean([ i == np.argmax(np.mean(audio_recognizer(memory), axis=0)) for i, memory in enumerate(audio_memories) ])*100)

                stride = scaled_audio.shape[0]/video_segment.shape[0]
                
                x = [ scaled_audio[scaled_audio.shape[0] - stride*video_segment.shape[0]::stride] ]
                y = [ video_segment ]
                video_producer.append(_train_network(x,y, output_dim=100))
                
                print 'Lessons learned in {} seconds'.format(time.time() - start_time)

                pushbutton['reset'] = True

                # if len(audio_memories) == 40:
                #     pickle.dump(audio_memories, open('counts.pickle','w'))
                #     print 'Data saved'

            if 'rmse' in pushbutton and len(audio):
                video_segment = np.array(list(video))
                audio_segment = np.array(list(audio))
                scaler = pp.MinMaxScaler()
                scaled_audio = scaler.fit_transform(audio_segment)

                plt.figure()
                plt.plot(chop(scaled_audio[:,idxs]))
                plt.draw()

                output = audio_recognizer(chop(scaled_audio[:,idxs]))
                plt.figure()
                plt.plot(output)
                plt.draw()

                winner = np.argmax(np.mean(output, axis=0))
                print 'WINNER NETWORK', winner

                sound = audio_producer[winner](scaled_audio)

                stride = audio_segment.shape[0]/video_segment.shape[0]

                projection = video_producer[winner](audio_segment[audio_segment.shape[0] - stride*video_segment.shape[0]::stride])

                for row in projection:
                    send_array(projector, row)

                for row in scaler.inverse_transform(sound):
                    send_array(speaker, row)

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


def chop(A, threshold=.05):
    right = A.shape[0]-1
    while A[right,0] < threshold:
        right -= 1
    left = 0 
    while A[left,0] < threshold:
        left += 1
    return A[left:right]

def gaussian_brain(host):
    name = 'BRAIN'+str(uuid1())
    
    print '{} connecting to {}'.format(name, host)

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
    
    audio_memories = []
    video_memories = []

    audio_recognizer = []
    audio_producer = []
    video_producer = []

    import matplotlib.pyplot as plt
    plt.ion()

    legends= ["level1", 
                                        "pitch1ptrack", 
                                        "pitch1pll", 
                                        "autocorr1", 
                                        "centroid1",
                                        "spread1", 
                                        "skewness1", 
                                        "kurtosis1", 
                                        "flatness1", 
                                        "crest1", 
                                        "flux1", 
                                        "epochSig1", 
                                        "epochRms1", 
                                        "epochZCcps1"]

    idxs = [0,6,7,8,9,12]
    #idxs = [0]
    #idxs = range(14) # Uncomment to include all parameters
    minlength = []
    
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

                start_time = time.time()
                audio_segment = np.array(list(audio))
                video_segment = np.array(list(video))
                scaler = pp.MinMaxScaler()
                scaled_audio = scaler.fit_transform(audio_segment)

                plt.figure()
                plt.plot(chop(scaled_audio[:,idxs]))
                plt.draw()

                audio_producer.append(_train_network([scaled_audio[:-1]], [scaled_audio[1:]]))

                audio_memories.append(chop(scaled_audio))#[:len(audio)/2,idxs])
                video_memories.append(video_segment)

                targets = range(len(audio_memories))

                minlength = min(map(lambda x: x.shape[0], audio_memories))
                print minlength

                train_data = [ np.ndarray.flatten(memory[:minlength,idxs]) for memory in audio_memories ]                
                                
                audio_recognizer = GaussianNB()
                audio_recognizer.fit(train_data, targets)
                
                print 'Testing recently learned audio segments: {}% correct'\
                    .format(np.mean([ i == audio_recognizer.predict(np.ndarray.flatten(memory[:minlength,idxs]))[0] for i, memory in enumerate(audio_memories) ])*100)

                stride = scaled_audio.shape[0]/video_segment.shape[0]
                
                x = [ scaled_audio[scaled_audio.shape[0] - stride*video_segment.shape[0]::stride] ]
                y = [ video_segment ]
                video_producer.append(_train_network(x,y, output_dim=100))
                
                print 'Lessons learned in {} seconds'.format(time.time() - start_time)

                pushbutton['reset'] = True

            if 'rmse' in pushbutton and len(audio):
                video_segment = np.array(list(video))
                audio_segment = np.array(list(audio))
                scaler = pp.MinMaxScaler()
                scaled_audio = scaler.fit_transform(audio_segment)

                plt.figure()
                plt.plot(scaled_audio[:minlength,idxs])
                plt.draw()
                
                selectah = np.ndarray.flatten(scaled_audio[:minlength,idxs])
                if selectah.shape[0] < minlength:
                    selectah = np.concatenate((selectah, np.zeros(minlength - selectah.shape[0],)))

                winner = audio_recognizer.predict(selectah)[0]
                print 'WINNER NETWORK', winner

                sound = audio_producer[winner](scaled_audio)

                stride = audio_segment.shape[0]/video_segment.shape[0]

                projection = video_producer[winner](audio_segment[audio_segment.shape[0] - stride*video_segment.shape[0]::stride])

                for row in projection:
                    send_array(projector, row)

                for row in scaler.inverse_transform(sound):
                    send_array(speaker, row)

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
                
                        
if __name__ == '__main__':
    #start_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    monolithic_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    #gaussian_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
