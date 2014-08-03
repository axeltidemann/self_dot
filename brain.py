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
from mlabwrap import init as matlab_init
from scipy.io import wavfile
from oct2py import Oct2Py
from scikits.samplerate import resample

from AI import learn, live, recognize, _train_network
from utils import filesize, recv_array, send_array, trim
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

                if audio_segment.shape[0] < 10:
                    continue

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

                # plt.figure()
                # plt.plot(audio_memories[-1])
                # plt.draw()

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

                # Parameters to network found by evolution: 58, 12, 98 for counts.pickle
                # 27 13 98 for dicks.pickle
                audio_recognizer = _train_network(audio_memories, targets, output_dim=270, leak_rate=.13, bias_scaling=.98) 

                print 'Testing recently learned audio segments: {}% correct'\
                    .format(np.mean([ i == np.argmax(np.mean(audio_recognizer(memory), axis=0)) for i, memory in enumerate(audio_memories) ])*100)

                stride = scaled_audio.shape[0]/video_segment.shape[0]
                
                x = [ scaled_audio[scaled_audio.shape[0] - stride*video_segment.shape[0]::stride] ]
                y = [ video_segment ]
                video_producer.append(_train_network(x,y, output_dim=100))
                
                print 'Lessons learned in {} seconds'.format(time.time() - start_time)

                pushbutton['reset'] = True

                # if len(audio_memories) == 40:
                #     pickle.dump(audio_memories, open('dicks.pickle','w'))
                #     print 'Data saved'

            if 'rmse' in pushbutton and len(audio):
                video_segment = np.array(list(video))
                audio_segment = np.array(list(audio))

                if audio_segment.shape[0] < 10:
                    continue

                scaler = pp.MinMaxScaler()
                scaled_audio = scaler.fit_transform(audio_segment)

                # plt.figure()
                # plt.plot(chop(scaled_audio[:,idxs]))
                # plt.draw()

                output = audio_recognizer(chop(scaled_audio[:,idxs]))
                # plt.figure()
                # plt.plot(output)
                # plt.draw()

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


def read_and_process(filename, matlab, plt=False):
    data, rate, nbits = matlab.wavread(filename, nout=3)
    decimator = 4
    data_resampled = resample(trim(data, threshold=.025), 1./decimator, 'sinc_best')
    data_resampled.shape = (data_resampled.shape[0], 1)
    matlab.wavwrite(data_resampled, rate/decimator, int(nbits), 'tmp.wav')
    carfac = matlab.CARFAC_hacking_axel('tmp.wav')

    if plt:
        plt.figure()
        plt.imshow(carfac, aspect='auto')
        plt.title(filename)
        plt.draw()

    return carfac.T
    
                
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
    wav_memories = []

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
    maxlen = []

    #matlab = matlab_init() #SPEED, BABY!
    matlab = Oct2Py() 
        
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

                print 'LEARN', pushbutton['wavfile']

                # Super ugly hack! Since Csound might not be finished writing to the file, we try to read it, and upon fail (i.e. it was not closed) we wait .1 seconds.
                while True:
                    try:
                        wavfile.read(pushbutton['wavfile'])
                        break
                    except:
                        time.sleep(.1)

                audio_memories.append(read_and_process(pushbutton['wavfile'], matlab, plt))
                wav_memories.append(pushbutton['wavfile'])

                maxlen = max([ memory.shape[0] for memory in audio_memories ])
                resampled_memories = [ resample(memory, float(maxlen)/memory.shape[0], 'sinc_best') for memory in audio_memories ]
                resampled_flattened_memories = [ np.ndarray.flatten(memory) for memory in resampled_memories ]
                
                audio_recognizer = GaussianNB()
                audio_recognizer.fit(resampled_flattened_memories, range(len(audio_memories)))
                
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

                test = read_and_process(pushbutton['wavfile'], matlab, plt)
                #test = matlab.resample(test, maxlen, test.shape[0])
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
                load_cns(pushbutton['load'], name)
                
                        
if __name__ == '__main__':
    #start_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    #monolithic_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
    gaussian_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
