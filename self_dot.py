#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import os
import multiprocessing as mp

import numpy as np
import csnd6

from AI import learn, respond
from IO import audio, video
from communication import receive as receive_messages
from utils import MyManager, MyDeque
       
class Controller:
    def __init__(self, sense, learn_state, respond_state):
        self.sense = sense
        self.learn_state = learn_state
        self.respond_state = respond_state
        
    def parse(self, message):
        print '[self.] received:', message

        if message == 'startrec':
            self.sense.value = 1

        if message == 'stoprec':
            self.sense.value = 0

        if message == 'learn':
            self.learn_state.put(True)

        if message == 'respond':
            self.respond_state.put(True)            
                        
if __name__ == '__main__':
    print 'MAIN PID', os.getpid()
    
    MyManager.register('deque', MyDeque)
    MyManager.register('list', list, proxytype=mp.managers.ListProxy)

    manager = MyManager()
    manager.start()

    brain = manager.list()
    mic = manager.deque()
    speaker = manager.deque()
    camera = manager.deque()
    projector = manager.deque()

    sense = mp.Value('i',0)
    learn_state = mp.Queue()
    respond_state = mp.Queue()

    controller = Controller(sense, learn_state, respond_state)

    mp.Process(target=audio, args=(sense, mic, speaker)).start()
    mp.Process(target=video, args=(sense, camera, projector)).start()
    mp.Process(target=learn, args=(learn_state, mic, camera, brain)).start()
    mp.Process(target=respond, args=(respond_state, mic, speaker, camera, projector, brain)).start()
    mp.Process(target=receive_messages, args=(controller.parse,)).start()
    
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
