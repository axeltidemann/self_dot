#!/usr/bin/python
# -*- coding: latin-1 -*-

''' Test suite for self.

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

from time import sleep

from IO import send

def play_sounds(secs):
    #soundfiles = ['dakata4.wav', 'fallpitch.wav', 'fox.wav', 'todidot.wav', 'washaboa.wav', 'count1.wav', 'count2.wav', 'tell1.wav']
    soundfiles = ['count1.wav', 'count2.wav', 'count3.wav', 'count4.wav']#, 'count5.wav', 'count6.wav', 'count7.wav', 'count8.wav']
    path = 'testsounds'

    for f in soundfiles:
        send('playfile {}/{}'.format(path, f))
        sleep(secs)


if __name__ == '__main__':

    send('calibrateAudio')
    raw_input('Press enter when calibrating is done')

    print 'Playing sounds [self.] will learn. Make gestures.'

    send('autolearn 1')
    play_sounds(8)
    send('autolearn 0')
    
    print 'Playing sounds [self.] will respond to. See if the gestures are the same.'

    send('autorespond 1')
    play_sounds(16)
    send('autorespond 0')
