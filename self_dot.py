#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import multiprocessing as mp
import time

import zmq
from zmq.utils.jsonapi import dumps

import IO
import utils
import brain
import robocontrol
import association

IDLE_SECONDS = 10
def idle(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, IO.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))

    face = context.socket(zmq.SUB)
    face.connect('tcp://{}:{}'.format(host, IO.FACE))
    face.setsockopt(zmq.SUBSCRIBE, b'')

    robocontrol = context.socket(zmq.PUSH)
    robocontrol.connect('tcp://localhost:{}'.format(IO.ROBO))

    poller = zmq.Poller()
    poller.register(stateQ, zmq.POLLIN)
    poller.register(face, zmq.POLLIN)
    
    state = stateQ.recv_json()

    state_time = time.time()
    face_time = time.time()
        
    while True:
        events = dict(poller.poll(timeout=IDLE_SECONDS*1000))
        if stateQ in events:
            state = stateQ.recv_json()
            state_time = time.time()

        if time.time() - state_time > IDLE_SECONDS and time.time() - face_time > IDLE_SECONDS:
            robocontrol.send_json([ 1, 'pan', .55 if np.random.rand() < 0.5 else .45]) 
            robocontrol.send_json([ 1, 'tilt', 45*np.randint(-5,5)])


        if face in events:
            new_face = utils.recv_array(face)
            face_time = time.time()

            
class Controller:
    def __init__(self, init_state):
        me = mp.current_process()
        print me.name, 'PID', me.pid

        self.state = init_state
        
        context = zmq.Context()
        
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind('tcp://*:{}'.format(IO.STATE))
        
        self.event = context.socket(zmq.PUB)
        self.event.bind('tcp://*:{}'.format(IO.EVENT))

        incoming = context.socket(zmq.PULL)
        incoming.bind('tcp://*:{}'.format(IO.EXTERNAL))

        while True:
            self.parse(incoming.recv_json())

    def parse(self, message):
        print '[self.] received:', message

        try:
            if 'calculate_cochlear' in message:
                _, wav_file = message.split()
                t0 = time.time()
                utils.write_cochlear(utils.wait_for_wav(wav_file))
                print 'Calculating cochlear neural activation patterns took {} seconds'.format(time.time() - t0)
            
            if 'register' in message and 'BRAIN' in message:
                _, name, free = message.split()
                self.state['brains'][name] = int(free)

            if 'fullscreen' in message:
                self.state['fullscreen'] = int(message[11:])

            if 'display2' in message:
                self.state['display2'] = int(message[9:])

            if message == 'startrec':
                self.state['record'] = True

            if message == 'stoprec':
                self.state['record'] = False

            if 'facerecognition' in message:
                _, value = message.split()
                self.state['facerecognition'] = value in ['True', '1']

            if 'showme' in message:
                self.event.send_json({ 'showme': message[7:] })

            if 'play_id' in message:
                self.event.send_json({ 'play_id': message[8:] })

            if 'testSentence' in message:
                self.event.send_json({ 'testSentence': message[13:] })

            if 'assoc_setParam' in message:
                self.event.send_json({ 'assoc_setParam': message[15:] })

            if 'respond_setParam' in message:
                self.event.send_json({ 'respond_setParam': message[17:] })

            if 'memoryRecording' in message:
                self.state['memoryRecording'] = message[16:] in ['True', '1']

            if 'roboActive' in message:
                self.state['roboActive'] = int(message[11:])

            if 'ambientSound' in message:
                self.state['ambientSound'] = int(message[13:])

            if 'decrement' in message:
                _, name = message.split()
                self.state['brains'][name] -= 1
                print '{} has now {} available slots'.format(name, self.state['brains'][name])

            if 'learnwav' in message:
                _, filename = message.split()
                # d = self.state['brains']
                # winner = max(d, key=d.get)
                # print '{} chosen to learn, has {} available slots'.format(winner, self.state['brains'][winner])
                # self.event.send_json({ 'learn': winner, 'filename': filename })
                self.event.send_json({ 'learn': True, 'filename': filename })

            if 'respondwav_single' in message:
                _, filename = message.split()
                self.event.send_json({ 'respond_single': True, 'filename': filename })

            if 'respondwav_sentence' in message:
                _, filename = message.split()
                self.event.send_json({ 'respond_sentence': True, 'filename': filename })

            if 'autolearn' in message:
                self.state['autolearn'] = message[10:] in ['True', '1']

            if 'autorespond_single' in message:
                self.state['autorespond_single'] = message[19:] in ['True', '1']

            if 'autorespond_sentence' in message:
                self.state['autorespond_sentence'] = message[21:] in ['True', '1']

            if 'inputLevel' in message:
                self.event.send_json({ 'inputLevel': message[11:] })

            if 'calibrateAudio' in message:
                self.event.send_json({ 'calibrateAudio': True })

            if 'csinstr' in message:
                self.event.send_json({ 'csinstr': message[8:] })
             
            if 'selfDucking' in message:
                self.event.send_json({ 'selfDucking': message[12:] })

            if 'zerochannels' in message:
                self.event.send_json({ 'zerochannels': message[13:] })

            if 'playfile' in message:
                self.event.send_json({ 'playfile': message[9:] })

            if 'selfvoice' in message:
                self.event.send_json({ 'selfvoice': message[10:] })

            if 'save' in message:
                self.event.send_json({ 'save': 'cns' if len(message) == 4 else message[5:] })

            if 'load' in message:
                self.event.send_json({ 'load': 'cns' if len(message) == 4 else message[5:] })

            self.publisher.send_json(self.state)

        except Exception as e:
            print utils.print_exception('Something went wrong when parsing the message - try again.')

if __name__ == '__main__':

    persistent_states = {'autolearn': False,
                         'autorespond_single': False,
                         'autorespond_sentence': False,
                         'brains': {},
                         'record': False,
                         'memoryRecording': False,
                         'roboActive': False,
                         'ambientSound': False,
                         'fullscreen': 0,
                         'display2': 0,
                         'facerecognition': False,}

    mp.Process(target=IO.audio, name='AUDIO').start() 
    mp.Process(target=IO.video, name='VIDEO').start()
    mp.Process(target=brain.face_extraction, args=('localhost',False,True,), name='FACE EXTRACTION').start()
    mp.Process(target=brain.respond, args=('localhost','localhost',), name='RESPONDER').start()
    mp.Process(target=brain.learn_audio, args=('localhost',), name='AUDIO LEARN').start()
    mp.Process(target=brain.learn_video, args=('localhost',), name='VIDEO LEARN').start()
    mp.Process(target=brain.learn_faces, args=('localhost',), name='FACES LEARN').start()
    mp.Process(target=brain.calculate_sai_video_marginals, args=('localhost',), name='SAI VIDEO CALCULATION').start()
    mp.Process(target=robocontrol.robocontrol, args=('localhost',), name='ROBOCONTROL').start()
    mp.Process(target=association.association, args=('localhost',), name='ASSOCIATION').start()
    mp.Process(target=Controller, args=(persistent_states,), name='CONTROLLER').start()
    mp.Process(target=idle, args=('localhost',), name='IDLER').start()
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
