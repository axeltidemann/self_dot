#!/usr/bin/python
# -*- coding: latin-1 -*-

#    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
#
#    This file is part of [self.]
#
#    [self.] is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 
#    as published by the Free Software Foundation.
#
#    [self.] is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with [self.].  If not, see <http://www.gnu.org/licenses/>.

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

    for f in glob.glob('testsounds/44100/*.wav')[:5]:
        send('playfile_input {}'.format(f))
        sleep(wav_duration(f) + wait_secs)

if __name__ == '__main__':

    for i in range(4):
        sleep(10)
        print 'Starting recording'
        send('startrec')
        sleep(3)
        send('stoprec')
        send('learnwav /Users/tidemann/Documents/NTNU/self_dot/testsounds/44100/count10_44100.wav')
        print 'Stopping recording'

    send('facerecognition 1')
    

    # send('calibrateAudio')
    # raw_input('Press enter when calibrating is done')

    # print 'Playing sounds [self.] will learn. Make gestures.'

    # send('autolearn 1')
    # send('memoryRecording 1')
    # play_sounds(10)
    # send('autolearn 0')
    # send('memoryRecording 0')

    # send('facerecognition 1')
    
    # print 'Playing sounds [self.] will respond to. See if the gestures are the same.'

    # send('autorespond 1')
    # play_sounds(10)
    # send('autorespond 0')
