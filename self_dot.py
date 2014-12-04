#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import multiprocessing as mp
import time
import cPickle as pickle

import zmq
from zmq.utils.jsonapi import dumps, loads
import numpy as np

import IO
import utils
import brain
import altbrain
import robocontrol
import association

def idle(host):
    context = zmq.Context()

    face = context.socket(zmq.SUB)
    face.connect('tcp://{}:{}'.format(host, IO.FACE))
    face.setsockopt(zmq.SUBSCRIBE, b'')

    robocontrol = context.socket(zmq.PUSH)
    robocontrol.connect('tcp://{}:{}'.format(host, IO.ROBO))

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, IO.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    state = stateQ.recv_json()

    poller = zmq.Poller()
    poller.register(face, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))

    face_timer = 0
    saysomething_timer = 0
    saysomething_interval = 10
    urge_to_say_something = 0
    
    def update_urge_to_say_something(urge_to_say_something, saysomething_interval):
        # linear increase over time for now
        # this could be more sophisticated
        # also including random impulses, emotions, etc
        return urge_to_say_something + saysomething_interval

    while True:
        events = dict(poller.poll(timeout=100))

        if face in events:
            new_face = utils.recv_array(face)
            face_timer = time.time()
                      
        if time.time() - face_timer > np.random.rand()*1.5 + 1:
            print '[self.] searches for a face'
            robocontrol.send_json([ 1, 'pan', (2*np.random.rand() -1)/10 ])
            robocontrol.send_json([ 1, 'tilt', (2*np.random.rand()-1)/10])
            face_timer = time.time()

        if stateQ in events:
            state = stateQ.recv_json()

        if not state['enable_say_something']:
            urge_to_say_something = 0

        if state['enable_say_something'] and not state['i_am_speaking'] and time.time() - saysomething_timer > saysomething_interval:
            urge_to_say_something = update_urge_to_say_something(urge_to_say_something, saysomething_interval)
            sender.send_json('urge_to_say_something {}'.format(urge_to_say_something))
            saysomething_timer = time.time()
            #print 'self idler trig and then disable say something'
            #sender.send_json('enable_say_something 0')
        
        
class Controller:
    def __init__(self, init_state, host):
        self.state = init_state
        
        context = zmq.Context()
        
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind('tcp://*:{}'.format(IO.STATE))
        
        self.event = context.socket(zmq.PUB)
        self.event.bind('tcp://*:{}'.format(IO.EVENT))

        snapshot = context.socket(zmq.ROUTER)
        snapshot.bind('tcp://*:{}'.format(IO.SNAPSHOT))

        self.association = context.socket(zmq.REQ)
        self.association.connect('tcp://{}:{}'.format(host, IO.ASSOCIATION))

        incoming = context.socket(zmq.PULL)
        incoming.bind('tcp://*:{}'.format(IO.EXTERNAL))

        poller = zmq.Poller()
        poller.register(incoming, zmq.POLLIN)
        poller.register(snapshot, zmq.POLLIN)

        while True:
            events = dict(poller.poll())
            
            if incoming in events:
                self.parse(incoming.recv_json())
                 
            if snapshot in events:
                address, _, message = snapshot.recv_multipart()
                snapshot.send_multipart([ address, 
                                          b'',
                                          dumps(self.state) ])

    def parse(self, message):
        print '[self.] received: {}'.format(message)

        black_list = []

        try:
            # if 'learnwav' in message or 'respondwav_single' in message or 'respondwav_sentence' in message:
            #     _, filename = message.split()
            #     if filename in black_list:
            #         print 'SKIPPING BAD FILE {}'.format(filename)
            #         return

            if message == 'dream':
                self.state['memoryRecording'] = False
                self.state['autorespond_sentence'] = False
                self.state['ambientSound'] = False
                self.state['autolearn'] = False
                self.state['autorespond_single'] = False
                self.state['_audioLearningStatus'] = False
                self.state['record'] = False
                self.publisher.send_json(self.state)

                self.event.send_json({'dream': True})

            if message == 'reboot':
                utils.reboot()

            if message == 'appendCurrentSettings' or message == 'popCurrentSettings':
                self.association.send_pyobj([message])
                self.association.recv_pyobj()
                
            if 'i_am_speaking' in message:
                _, value = message.split()
                self.state['i_am_speaking'] = value in ['True', '1']

            if 'enable_say_something' in message:
                _, value = message.split()
                self.state['enable_say_something'] = value in ['True', '1']
            
            if 'last_segment_ids' in message:
                the_ids = message[17:]
                self.event.send_json({'last_segment_ids': loads(the_ids) })
                
            if 'last_most_significant_audio_id' in message:
                audio_id = message[31:]
                self.event.send_json({'last_most_significant_audio_id': audio_id })
            
            if message == 'clear play_events':
                self.event.send_json({'clear play_events' : 'True'})

            if 'calculate_cochlear' in message:
                _, wav_file = message.split()
                t0 = time.time()
                try:
                    brain.cochlear(utils.wait_for_wav(wav_file), stride=IO.NAP_STRIDE, rate=IO.NAP_RATE)
                except:
                    print 'SHOULD {} BE BLACKLISTED?'.format(wav_file)
                    black_list.append(wav_file)
                print 'Calculating cochlear neural activation patterns took {} seconds'.format(time.time() - t0)
            
            if message == 'evolve':
                self.state['memoryRecording'] = False
                self.state['autorespond_sentence'] = False
                self.state['autolearn'] = False
                self.state['autorespond_single'] = False
                self.state['_audioLearningStatus'] = False
                self.state['record'] = False
                self.publisher.send_json(self.state)
                
                self.association.send_pyobj(['evolve'])
                self.association.recv_pyobj()

            if 'register' in message and 'BRAIN' in message:
                _, name, free = message.split()
                self.state['brains'][name] = int(free)

            if 'fullscreen' in message:
                _, value = message.split()
                self.event.send_json({ 'fullscreen': value in ['True', '1'] })

            if 'display2' in message:
                _, value = message.split()
                self.event.send_json({ 'display2': value in ['True', '1'] })

            if message == 'startrec':
                self.state['record'] = True

            if message == 'stoprec':
                self.state['record'] = False

            if 'facerecognition' in message:
                _, value = message.split()
                self.state['facerecognition'] = value in ['True', '1']

            if 'print_me' in message:
                self.event.send_json({ 'print_me': message[7:] })

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
               
            if '_audioLearningStatus' in message:
                self.state['_audioLearningStatus'] = message[21:] in ['True', '1']

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
                self.event.send_json({ 'learn': True, 'filename': filename })

            if 'respondwav_single' in message:
                _, filename = message.split()
                self.event.send_json({ 'respond_single': True, 'filename': filename })

            if 'respondwav_sentence' in message:
                _, filename = message.split()
                self.event.send_json({ 'respond_sentence': True, 'filename': filename })

            if 'play_sentence' in message:
                print 'playSentence', message
                sentence = message[13:]
                self.event.send_json({ 'play_sentence':True, 'sentence': sentence })

            if 'rhyme' in message:
                _, value = message.split()
                self.event.send_json({'rhyme': value == 'True'})

            if 'urge_to_say_something' in message:
                _, value = message.split()
                self.event.send_json({'urge_to_say_something': value})

            if 'autolearn' in message:
                self.state['autolearn'] = message[10:] in ['True', '1']

            if 'autorespond_single' in message:
                self.state['autorespond_single'] = message[19:] in ['True', '1']

            if 'autorespond_sentence' in message:
                self.state['autorespond_sentence'] = message[21:] in ['True', '1']

            if 'inputLevel' in message:
                self.event.send_json({ 'inputLevel': message[11:] })

            if 'calibrateEq' in message:
                self.event.send_json({ 'calibrateEq': True })

            if 'calibrateAudio' in message:
                latency_ok = False
                try:
                    lat = open('roundtrip_latency.txt', 'r')
                    latency = float(lat.readline())
                    self.event.send_json({ 'setLatency': latency })
                    latency_ok = True
                except Exception, e:
                    print 'Something went wrong when reading latency from file.', e
                    self.event.send_json({ 'calibrateAudio': True })
                if latency_ok:
                    self.event.send_json({ 'calibrateNoiseFloor': True }) 
                if 'calibrateAudio memoryRecording' in message:
                    self.state['memoryRecording'] = True

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
                self.event.send_json({ 'save': utils.brain_name() if len(message) == 4 else message[5:] })

            if 'load' in message:
                if len(message) == 4:
                    brain_name = utils.find_last_valid_brain()
                else:
                    _, brain_name = message.split()
                if brain_name:
                    self.event.send_json({ 'load': brain_name })

            self.publisher.send_json(self.state)

        except Exception as e:
            utils.print_exception('Something went wrong when parsing the message - try again.')

if __name__ == '__main__':

    persistent_states = {'autolearn': False,
                         'autorespond_single': False,
                         'autorespond_sentence': False,
                         'brains': {},
                         'record': False,
                         'memoryRecording': False,
                         'roboActive': False,
                         '_audioLearningStatus': False, 
                         'enable_say_something': False,
                         'i_am_speaking': False,
                         'ambientSound': False,
                         'fullscreen': 0,
                         'display2': 0,
                         'facerecognition': False,}

    me = mp.current_process()
    print 'SELF MAIN PID', me.pid

    utils.MyProcess(target=IO.audio, name='AUDIO').start() 
    utils.MyProcess(target=IO.video, name='VIDEO').start()
    utils.MyProcess(target=brain.face_extraction, args=('localhost',False,True,), name='FACE EXTRACTION').start()
    #utils.MyProcess(target=brain.respond, args=('localhost','localhost',True), name='RESPONDER').start()
    utils.MyProcess(target=altbrain.respond, args=('localhost','localhost',False,), name='RESPONDER').start()
    #utils.MyProcess(target=brain.learn_audio, args=('localhost',True), name='AUDIO LEARN').start()
    utils.MyProcess(target=altbrain.learn_audio, args=('localhost',True), name='AUDIO LEARN').start()
    utils.MyProcess(target=brain.learn_video, args=('localhost',), name='VIDEO LEARN').start()
    utils.MyProcess(target=brain.learn_faces, args=('localhost',), name='FACES LEARN').start()
    utils.MyProcess(target=robocontrol.robocontrol, args=('localhost',), name='ROBOCONTROL').start()
    utils.MyProcess(target=association.association, args=('localhost',), name='ASSOCIATION').start()
    utils.MyProcess(target=brain.cognition, args=('localhost',), name='COGNITION').start()
    utils.MyProcess(target=utils.scheduler, args=('localhost',), name='SCHEDULER').start()
    utils.MyProcess(target=Controller, args=(persistent_states,'localhost',), name='CONTROLLER').start()
    utils.MyProcess(target=idle, args=('localhost',), name='IDLER').start()
    utils.MyProcess(target=utils.counter, args=('localhost',), name='COUNTER').start()
    utils.MyProcess(target=utils.sentinel, args=('localhost',), name='SENTINEL').start()
    utils.daily_routine('localhost')
