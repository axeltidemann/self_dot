import time
import multiprocessing as mp
import cPickle as pickle
from uuid import uuid1
from collections import deque

import zmq
from sklearn import preprocessing as pp
import numpy as np

from utils import signal_rmse, sleep, filesize, send_array, recv_array
from IO import MIC, SPEAKER, CAMERA, PROJECTOR, STATE, SNAPSHOT, EVENT, EXTERNAL


def learn(audio_data, video_data, host):

    # Create recognizer and producer of audio and video, so the logic of this mix can be done in brain.py
    from esn import ACDCESN
    import Oger
    import mdp
    
    print '[self.] learns', 
    start_time = time.time()

    scaler = pp.MinMaxScaler() 
    scaled_data = scaler.fit_transform(audio_data)

    mdp.numx.random.seed(7)
    
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                              leak_rate=0.9, 
                                              bias_scaling=.2, 
                                              reset_states=False)
    readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
    audio_net = mdp.hinet.FlowNode(reservoir + readout)

    x = scaled_data[:-1]
    y = scaled_data[1:]

    audio_net.train(x,y)

    audio_video_net = ACDCESN(hidden_nodes=100,
                              leak_rate=0.9,
                              bias_scaling=.2,
                              reset_states=True,
                              use_pinv=True)

    # Video is sampled at a much lower frequency than audio.
    stride = audio_data.shape[0]/video_data.shape[0]

    x = scaled_data[scaled_data.shape[0] - stride*video_data.shape[0]::stride]
    y = video_data
    
    audio_video_net.train(x,y)

    print 'in {} seconds'.format(time.time() - start_time)
    live(audio_net, audio_video_net, scaler, host)

    
def live(audio_net, audio_video_net, scaler, host):
    import Oger

    me = mp.current_process()
    print me.name, 'PID', me.pid

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

    snapshot = context.socket(zmq.REQ)
    snapshot.connect('tcp://{}:{}'.format(host, SNAPSHOT))
    snapshot.send(b'Send me the state, please')
    state = snapshot.recv_json()

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, EXTERNAL))
    sender.send_json('register {}'.format(me.name))

    poller = zmq.Poller()
    poller.register(mic, zmq.POLLIN)
    poller.register(camera, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    error = deque()
    previous_prediction = []
    # Approximately 10 seconds of audio/video
    audio = deque(maxlen=3400)
    video = deque(maxlen=80)
    while True:
        events = dict(poller.poll())

        if stateQ in events:
            state = stateQ.recv_json()

        if mic in events:
            new_audio = np.atleast_2d(recv_array(mic))
            scaled_signals = scaler.transform(new_audio)
            if len(previous_prediction):
                error.append(scaled_signals.flatten() - previous_prediction.flatten())
            previous_prediction = audio_net(scaled_signals)
            if state['record']:
                audio.append(np.ndarray.flatten(scaled_signals))

        if camera in events:
            new_video = recv_array(camera)
            if state['record']:
                video.append(new_video)

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'reset' in pushbutton:
                error.clear()
                audio.clear()
                video.clear()

            if 'rmse' in pushbutton:
                rmse = np.sqrt((np.array(list(error)).flatten() ** 2).mean())
                sender.send_json('{} RMSE {}'.format(me.name, rmse))
                
            if 'respond' in pushbutton and pushbutton['respond'] == me.name:
                audio_data = np.array(list(audio))
                video_data = np.array(list(video))

                print '{} chosen to respond. Audio data: {} Video data: {}'.format(me.name, audio_data.shape, video_data.shape)

                sound = audio_net(audio_data)

                stride = audio_data.shape[0]/video_data.shape[0]
                projection = audio_video_net(audio_data[audio_data.shape[0] - stride*video_data.shape[0]::stride]) 

                # DREAM MODE: You can train a network with zero audio input -> video output, and use this
                # to recreate the original training sequence with scary accuracy...

                # Ordering to try to align video/sound 
                for row in projection:
                    send_array(projector, row)

                for row in scaler.inverse_transform(sound):
                    send_array(speaker, row)





def old_live(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler):
    me = mp.current_process()
    me.name = 'NEURAL NETWORK:'+me.name[10:]
    print me.name, 'PID', me.pid

    saved = False
    state[me.name] = 'init'
    
    rmse = []
    while sleep(.1):
        if state[me.name] == 'RESET':
            rmse = []
        
        if state['record']:
            try:
                rmse.append(signal_rmse(audio_net, scaler, mic.latest(me.name)))
                state[me.name] = np.mean(rmse)
                print me.name, 'RMSE', state[me.name]
            except:
                print me.name, 'HICKUP!'

        if state['respond'] == me.name:
            print me.name, 'chosen to respond'
            audio_data = mic.array()
            video_data = camera.array()

            mic.clear()
            camera.clear()

            scaled_data = scaler.transform(audio_data)
            sound = audio_net(scaled_data)

            stride = audio_data.shape[0]/video_data.shape[0]
            projection = audio_video_net(scaled_data[scaled_data.shape[0] - stride*video_data.shape[0]::stride]) 

            # DREAM MODE: You can train a network with zero audio input -> video output, and use this
            # to recreate the original training sequence with scary accuracy...

            # Ordering to try to align video/sound 
            for row in projection:
                projector.append(row)

            for row in scaler.inverse_transform(sound):
                speaker.append(row)

            state['respond'] = False

        if not saved and state['save']:
            filename = state['save'] + str(uuid1())
            pickle.dump((audio_net, audio_video_net, scaler), file(filename, 'w'))
            print '{} saved as file {} ({})'.format(me.name, filename, filesize(filename))
            saved = True
            




def old_learn(audio_data, video_data, markers=None):
    from esn import ACDCESN
    import Oger
    import mdp
    
    print '[self.] learns', 
    start_time = time.time()

    audio_data = mic.array()
    video_data = camera.array()

    micmarker = mic.get_mark()
    videomarker = camera.get_mark()

    mic.clear()
    camera.clear()
    
    scaler = pp.MinMaxScaler() 
    scaled_data = scaler.fit_transform(audio_data)

    mdp.numx.random.seed(7)
    
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                              leak_rate=0.9, 
                                              bias_scaling=.2, 
                                              reset_states=False)
    readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
    audio_net = mdp.hinet.FlowNode(reservoir + readout)

    if micmarker:
        x = scaled_data[:micmarker]
        y = scaled_data[micmarker:]

        row_diff = x.shape[0] - y.shape[0]
        if row_diff < 0:
            x = np.vstack([ x, np.zeros((-row_diff, x.shape[1])) ])
        elif row_diff > 0:
            y = np.vstack([ y, np.zeros((row_diff, y.shape[1])) ])
        print 'a heteroassociation',
    else:
        x = scaled_data[:-1]
        y = scaled_data[1:]
        print 'an imitation', 
    
    audio_net.train(x,y)

    audio_video_net = ACDCESN(hidden_nodes=100,
                              leak_rate=0.9,
                              bias_scaling=.2,
                              reset_states=True,
                              use_pinv=True)

    # Video is sampled at a much lower frequency than audio.
    stride = audio_data.shape[0]/video_data.shape[0]

    if micmarker:
        x = scaled_data[scaled_data.shape[0] - stride*video_data.shape[0]:micmarker:stride]
        y = video_data[videomarker:]

        row_diff = x.shape[0] - y.shape[0]
        if row_diff < 0:
            x = np.vstack([ x, np.zeros((-row_diff, x.shape[1])) ])
        elif row_diff > 0:
            y = np.vstack([ y, np.zeros((row_diff, y.shape[1])) ])
    else:
        x = scaled_data[scaled_data.shape[0] - stride*video_data.shape[0]::stride]
        y = video_data
    
    audio_video_net.train(x,y)

    print 'in {} seconds'.format(time.time() - start_time)

    live(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler)

