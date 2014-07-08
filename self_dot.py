#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import multiprocessing as mp
import zmq
from zmq.utils.jsonapi import dumps

from AI import learn
from IO import audio, video, STATE, EXTERNAL, SNAPSHOT, EVENT # load_cns
from utils import MyDeque, reset_rmses, find_winner
       
class Controller:
    def __init__(self, init_state):
        me = mp.current_process()
        print me.name, 'PID', me.pid

        self.state = init_state
        context = zmq.Context()
        self.publisher = context.socket(zmq.PUB)
        self.publisher.bind('tcp://*:{}'.format(STATE))
        
        self.event = context.socket(zmq.PUB)
        self.event.bind('tcp://*:{}'.format(EVENT))

        subscriber = context.socket(zmq.PULL)
        subscriber.bind('tcp://*:{}'.format(EXTERNAL))

        snapshot = context.socket(zmq.ROUTER)
        snapshot.bind('tcp://*:{}'.format(SNAPSHOT))

        print 'Communication channel listening on port {}'.format(EXTERNAL)

        poller = zmq.Poller()
        poller.register(subscriber, zmq.POLLIN)
        poller.register(snapshot, zmq.POLLIN)

        while True:
            events = dict(poller.poll())

            if subscriber in events:
                self.parse(subscriber.recv_json())
                
            if snapshot in events:
                address, _, message = snapshot.recv_multipart()
                snapshot.send_multipart([ address, 
                                          b'',
                                          dumps(self.state) ])
                

    def parse(self, message):
        print '[self.] received:', message

        try:
            if 'register' in message and 'BRAIN' in message:
                _, name, free = message.split()
                self.state['brains'][name] = int(free)

            if 'register' in message and 'NEURALNETWORK' in message:
                _, name = message.split()
                self.state['nets'].append(name)

            if 'RMSE' in message:
                name, _, rmse = message.split()
                self.state['RMSEs'][name] = float(rmse)
                if len(self.state['RMSEs']) == len(self.state['nets']):
                    d = self.state['RMSEs']
                    winner = min(d, key=d.get)
                    self.event.send_json({ 'respond': winner })
                    self.event.send_json({ 'reset': True })
                    self.state['RMSEs'] = {}
            
            if message == 'startrec':
                self.state['record'] = True

            if message == 'stoprec':
                self.state['record'] = False

            if message == 'learn':
                d = self.state['brains']
                winner = max(d, key=d.get)
                print '{} chosen to learn, has {} available slots'.format(winner, self.state['brains'][winner])
                self.state['brains'][winner] -= 1
                self.event.send_json({ 'learn': winner })

            if message == 'respond':
                self.event.send_json({ 'rmse': True })

            if message == 'setmarker':
                print 'EVALUATE USAGE OF THESE'

            if 'autolearn' in message:
                self.state['autolearn'] = message[10:] in ['True', '1']

            if 'autorespond' in message:
                self.state['autorespond'] = message[12:] in ['True', '1']

            if 'inputLevel' in message:
                self.event.send_json({ 'inputLevel': message[11:] })

            if 'calibrateAudio' in message:
                self.event.send_json({ 'calibrateAudio': True })

            if 'csinstr' in message:
                self.event.send_json({ 'csinstr': message[8:] })
             
            if 'zerochannels' in message:
                self.event.send_json({ 'zerochannels': message[13:] })

            if 'playfile' in message:
                self.event.send_json({ 'playfile': message[9:] })

            if 'selfvoice' in message:
                self.event.send_json({ 'selfvoice': message[10:] })

            if 'save' in message:
                self.event.send_json({ 'save': 'brain' if len(message) == 4 else message[5:] })

            # if 'load' in message:
            #     self.state['load'] = message[5:]
            #     load_cns(self.state, self.mic, self.speaker, self.camera, self.projector)

            self.publisher.send_json(self.state)

        except Exception as e:
            print 'Something went wrong when parsing the message - try again.'
            print e

if __name__ == '__main__':

    persistent_states = {'autolearn': False,
                         'autorespond': False,
                         'brains': {},
                         'nets': [],
                         'RMSEs': {},
                         'record': False,}

    mp.Process(target=audio, name='AUDIO').start() 
    mp.Process(target=video, name='VIDEO').start()
    mp.Process(target=Controller, args=(persistent_states,), name='CONTROLLER').start()

    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
