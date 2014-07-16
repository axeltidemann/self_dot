import time
import multiprocessing as mp
import cPickle as pickle
from uuid import uuid1
from collections import deque

import zmq
from sklearn import preprocessing as pp
import numpy as np

from utils import filesize, send_array, recv_array
from IO import MIC, SPEAKER, CAMERA, PROJECTOR, STATE, SNAPSHOT, EVENT, EXTERNAL

def _train_network(x, y, output_dim=100, leak_rate=.9, bias_scaling=.2, reset_states=False, use_pinv=True):
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


def learn(audio_in, audio_out, video_in, video_out, host):
    print '[self.] learns', 
    start_time = time.time()

    scaler = pp.MinMaxScaler() 
    scaled_audio = scaler.fit_transform(np.vstack([ audio_in, audio_out ]))
    scaled_audio_in = scaled_audio[:len(audio_in)]
    scaled_audio_out = scaled_audio[len(audio_in):]

    x = scaled_audio_in[:-1]
    y = scaled_audio_in[1:]
    audio_recognizer = _train_network(x, y)

    row_diff = audio_in.shape[0] - audio_out.shape[0]
    if row_diff < 0:
        scaled_audio_in = np.vstack([ scaled_audio_in, np.zeros((-row_diff, scaled_audio_in.shape[1])) ]) # Zeros because of level
    elif row_diff > 0:
        scaled_audio_in = scaled_audio_in[:len(scaled_audio_out)]

    x = scaled_audio_in[:-1]
    y = scaled_audio_out[1:]
    audio_producer = _train_network(x, y)
    audio_producer.length = audio_out.shape[0]

    # Video is sampled at a much lower frequency than audio.
    stride = audio_out.shape[0]/video_out.shape[0]
    
    x = scaled_audio_in[scaled_audio_in.shape[0] - stride*video_out.shape[0]::stride]
    y = video_out
    
    audio2video = _train_network(x, y)
    audio2video.length = video_out.shape[0]

    print 'in {} seconds'.format(time.time() - start_time)
    live(audio_recognizer, audio_producer, audio2video, scaler, host)

    
def live(audio_recognizer, audio_producer, audio2video, scaler, host):
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
            previous_prediction = audio_recognizer(scaled_signals)
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

                row_diff = audio_data.shape[0] - audio_producer.length
                if row_diff < 0:
                    audio_data = np.vstack([ audio_data, np.zeros((-row_diff, audio_data.shape[1])) ])
                else:
                    audio_data = audio_data[:audio_producer.length]

                sound = audio_producer(audio_data)
                
                stride = audio_producer.length/audio2video.length
                projection = audio2video(audio_data[audio_data.shape[0] - stride*audio2video.length::stride])

                # DREAM MODE: You can train a network with zero audio input -> video output, and use this
                # to recreate the original training sequence with scary accuracy...

                # Ordering to try to align video/sound 
                for row in projection:
                    send_array(projector, row)

                for row in scaler.inverse_transform(sound):
                    send_array(speaker, row)

            if 'save' in pushbutton:
                filename = '{}.{}'.format(pushbutton['save'], me.name)
                pickle.dump((audio_recognizer, audio_producer, audio2video, scaler, host), file(filename, 'w'))
                print '{} saved as file {} ({})'.format(me.name, filename, filesize(filename))
