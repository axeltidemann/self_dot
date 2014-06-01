import time
import multiprocessing as mp
import cPickle as pickle
from uuid import uuid4

from sklearn import preprocessing as pp
import numpy as np

from utils import net_rmse, sleep, filesize

def find_winner(state):
    min_rmse = np.inf
    winner = ''
    for key in state.keys():
        if key.startswith('NEURAL') and np.mean(state[key][-10:]) < min_rmse: #THE SELECTION OF LATEST 10 ELEMENTS SHOULD BE FIXED
            min_rmse = np.mean(state[key])
            winner = key
    state['respond'] = winner

    
def learn(state, mic, speaker, camera, projector):
    from esn import ACDCESN
    import Oger
    import mdp
    
    print '[self.] learns...', 
    start_time = time.time()

    audio_data = mic.array()
    video_data = camera.array()

    mic.clear()
    camera.clear()

    scaler = pp.MinMaxScaler() 
    scaled_data = scaler.fit_transform(audio_data)

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                              leak_rate=0.8, 
                                              bias_scaling=.2, 
                                              reset_states=False)
    readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
    audio_net = mdp.hinet.FlowNode(reservoir + readout)

    x = scaled_data[:-1]
    y = scaled_data[1:]

    audio_net.train(x,y)

    audio_video_net = ACDCESN(hidden_nodes=500,
                              leak_rate=0.8,
                              bias_scaling=.2,
                              reset_states=True,
                              use_pinv=True)

    # Video is sampled at a much lower frequency than audio.
    stride = audio_data.shape[0]/video_data.shape[0]
    x = scaled_data[scaled_data.shape[0] - stride*video_data.shape[0]::stride]
    y = video_data

    audio_video_net.train(x,y)

    print 'finished learning audio-video association in {} seconds'.format(time.time() - start_time)

    live(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler)

    
def live(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler):
    me = mp.current_process()
    me.name = 'NEURAL NETWORK:'+me.name[10:]
    print me.name, 'PID', me.pid

    saved = False

    rmse = []
    while sleep(.1):
        if state['record']:
            try:
                rmse = rmse + net_rmse([ (audio_net, scaler) ], mic.latest(me.name))
                state[me.name] = rmse
                print me.name, 'RMSE', state[me.name][-1]
            except:
                print 'Audio buffer was emptied before recognize could finish.'

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



