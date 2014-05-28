#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import os
import multiprocessing as mp
from uuid import uuid4

from AI import learn, respond, recognize
from IO import audio, video
from communication import receive as receive_messages
from utils import MyManager, MyDeque
       
class Controller:
    def __init__(self, state):
        self.state = state
        
    def parse(self, message):
        print '[self.] received:', message

        if message == 'startrec':
            self.state['record'] = True

        if message == 'stoprec':
            self.state['record'] = False

        if message == 'learn':
            self.state['learn'] = True

        if message == 'respond':
            self.state['respond'] = True

        if 'autolearn' in message:
            try: 
                mess = message[10:]
                if (mess == 'True') or (mess == '1'):
                    self.state['autolearn'] = True
                elif (mess == 'False') or (mess == '0'):
                    self.state['autolearn'] = False
                else:
                    print 'unknown autolearn mode', mess
            except: 
                print 'invalid autolearn mode', message
            print 'autolearn set to:', self.state['autolearn']

        if 'autorespond' in message:
            try: 
                mess = message[12:]
                if (mess == 'True') or (mess == '1'):
                    self.state['autorespond'] = True
                elif (mess == 'False') or (mess == '0'):
                    self.state['autorespond'] = False
                else:
                    print 'unknown autorespond mode', mess
            except: 
                print 'invalid autorespond mode', message
            print 'autorespond set to:', self.state['autorespond']

        if 'inputLevel' in message:
            self.state['inputLevel'] = message[11:]

        if 'csinstr' in message:
            self.state['csinstr'] = message[8:]

        if 'playfile' in message:
            self.state['playfile'] = message[9:]

        if 'selfvoice' in message:
            self.state['selfvoice'] = message[10:]

        if 'save' in message:
            self.state['save'] = 'brain' + str(uuid4()) if len(message) == 4 else message[5:]
                        
        if 'load' in message:
            self.state['load'] = message[5:]

if __name__ == '__main__':
    print 'MAIN PID', os.getpid()
    
    MyManager.register('deque', MyDeque)

    manager = MyManager()
    manager.start()

    brain = manager.list()
    mic = manager.deque()
    speaker = manager.deque()
    camera = manager.deque()
    projector = manager.deque()

    state = manager.dict({'record': False,
                          'learn': False,
                          'respond': False,
                          'autolearn': False,
                          'autorespond': False,
                          'inputLevel': False, 
                          'csinstr': False, 
                          'playfile': False, 
                          'selfvoice':False,
                          'save': False, 
                          'load': False, 
                          'rmse': []}) 

    controller = Controller(state)

    mp.Process(target=audio, args=(state, mic, speaker)).start()
    mp.Process(target=video, args=(state, camera, projector)).start()
    mp.Process(target=learn, args=(state, mic, camera, brain)).start()
    mp.Process(target=respond, args=(state, mic, speaker, camera, projector, brain)).start()
    mp.Process(target=recognize, args=(state, mic, camera, brain)).start()
    mp.Process(target=receive_messages, args=(controller.parse,)).start()
    
    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
