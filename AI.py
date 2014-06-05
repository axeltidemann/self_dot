import time
import multiprocessing as mp
import cPickle as pickle
from uuid import uuid4

from sklearn import preprocessing as pp
import numpy as np

from utils import signal_rmse, sleep, filesize


def learn(state, mic, speaker, camera, projector):
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

    
def live(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler):
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
            filename = state['save'] + str(uuid4())
            pickle.dump((audio_net, audio_video_net, scaler), file(filename, 'w'))
            print '{} saved as file {} ({})'.format(me.name, filename, filesize(filename))
            saved = True
            
