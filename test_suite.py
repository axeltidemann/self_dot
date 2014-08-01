#!/usr/bin/python
# -*- coding: latin-1 -*-

''' Test suite for self.

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

from time import sleep
import glob

from IO import send
from utils import wav_duration

def play_sounds(wait_secs):

    for f in glob.glob('testsounds/*.wav'):
        send('playfile {}'.format(f))
        sleep(wav_duration(f) + wait_secs)

if __name__ == '__main__':

    send('calibrateAudio')
    raw_input('Press enter when calibrating is done')

    print 'Playing sounds [self.] will learn. Make gestures.'

    send('autolearn 1')
    play_sounds(7)
    send('autolearn 0')
    
    print 'Playing sounds [self.] will respond to. See if the gestures are the same.'

    send('autorespond 1')
    play_sounds(10)
    send('autorespond 0')
