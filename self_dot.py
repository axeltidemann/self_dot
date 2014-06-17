#!/usr/bin/python
# -*- coding: latin-1 -*-

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

import multiprocessing as mp

from AI import learn
from IO import audio, video, load_cns
from communication import receive as receive_messages
from utils import MyDeque, reset_rmses, find_winner
       
class Controller:
    def __init__(self, state, mic, speaker, camera, projector):
        self.state = state
        self.mic = mic
        self.speaker = speaker
        self.camera = camera
        self.projector = projector
        
    def parse(self, message):
        print '[self.] received:', message

        try:
            if message == 'startrec':
                reset_rmses(self.state)
                self.state['record'] = True

            if message == 'stoprec':
                self.state['record'] = False

            if message == 'learn':
                mp.Process(target=learn,
                            args=(self.state, self.mic, self.speaker, self.camera, self.projector)).start()

            if message == 'respond':
                find_winner(self.state)

            if message == 'setmarker':
                self.mic.set_mark()
                self.camera.set_mark()

            if 'autolearn' in message:
                self.state['autolearn'] = message[10:] in ['True', '1']

            if 'autorespond' in message:
                self.state['autorespond'] = message[12:] in ['True', '1']

            if 'inputLevel' in message:
                self.state['inputLevel'] = message[11:]

            if 'calibrateAudio' in message:
                self.state['calibrateAudio'] = True

            if 'csinstr' in message:
                self.state['csinstr'] = message[8:]
             
            if 'zerochannels' in message:
                self.state['zerochannels'] = message[13:]

            if 'playfile' in message:
                self.state['playfile'] = message[9:]

            if 'selfvoice' in message:
                self.state['selfvoice'] = message[10:]

            if 'save' in message:
                self.state['save'] = 'brain' if len(message) == 4 else message[5:]

            if 'load' in message:
                self.state['load'] = message[5:]
                load_cns(self.state, self.mic, self.speaker, self.camera, self.projector)

        except Exception as e:
            print 'Something went wrong when parsing the message - try again.'
            print e

if __name__ == '__main__':
    me = mp.current_process()
    print me.name, 'PID', me.pid

    class LocalManager(SyncManager):
        pass
        
    LocalManager.register('deque', MyDeque)

    manager = LocalManager()
    manager.start()

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
                          'calibrateAudio': False, 
                          'csinstr': False, 
                          'zerochannels': False, 
                          'playfile': False, 
                          'selfvoice':False,
                          'save': False, 
                          'load': False,
                      }) 

    controller = Controller(state, mic, speaker, camera, projector)

    mp.Process(target=audio, args=(state, mic, speaker,)).start() 
    mp.Process(target=video, args=(state, camera, projector)).start()
    mp.Process(target=receive_messages, args=(controller.parse,)).start()
    
    class ServerManager(SyncManager):
        pass
    
    ServerManager.register('get_state', state)
    server_manager = ServerManager(address=('', 7777), authkey='tullball')
    server_manager.start()

    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), mp.active_children())
