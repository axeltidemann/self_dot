#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import zmq
import IO
import time
import random

# connect to arduino
import serial
import numpy as np
import utils

class NoSerial:
    def readline(self):
        pass
    def read(self):
        pass
    def write(self, arg):
        pass
    
try:
    ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0)
    serialAvailable = 1
except:
    ser = NoSerial()
    serialAvailable = 0
    print '*****************************************'
    print 'Robot not connected or comunication error'
    print '*****************************************'

connected = False
pan1 = 60
tilt1 = 20
pan2 = 30
tilt2 = 45
if serialAvailable:
    while not connected:
        serin = ser.read()
        connected = True
        print 'Robot connected'
        
    time.sleep(2)
    print 'set initial position'
    ser.write('t %03dn'%tilt1)
    time.sleep(.2)
    print '...' 
    ser.write('p %03dn'%pan1)
    time.sleep(.2)
    print '...' 
    ser.write('r %03dn'%tilt1)
    time.sleep(.2)
    print '...' 
    ser.write('o %03dn'%pan1)
    print '...' 
    print 'initial position is set'
    time.sleep(1)

def robocontrol(host):
    context = zmq.Context()

    robo = context.socket(zmq.PULL)
    robo.bind('tcp://*:{}'.format(IO.ROBO))

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))

    timeStamp = time.time()
    global pan1, tilt1, pan2, tilt2
    time_reset_memrec = time.time()
    memrec_turnon = False
    memrec_turnoff = False
    search_pan_index = 0
    search_tilt_index = 0
    search_tiltpos = [20, 10, 30, 20, 5, 15, 25, 35, 20, 25, 15, 5, 0, 10, 20, 30, 40, 30]

    gain = lambda x: .5 + np.exp(-x) # Exponential decay, the constant is percentwise increase.
    i = 0
    while True:
    	time.sleep(.05)
        robohead,axis,value = robo.recv_json()
        if memrec_turnon and (time.time()-time_reset_memrec > 1.7):
            sender.send_json('memoryRecording 1')
            memrec_turnon = False
            print 'ROBOCTRL TURN ON MEMREC AFTER HEAD SPIN'
        if robohead == 1:
            if axis == 'transient':
                print 'ROBOCTRL TRANSIENT, GAINING OUTPUT {}'.format(gain(0))
                i = 0

            if axis == 'pan': #when we have adjustment due to sound input...
                search_pan_index = 0 #do only fine adjustment
            if axis == 'search': #searching for a face
                pan1 += (random.random()-0.5)*20
                search_pan_index += 1
                search_pan_index %= 1000
                if search_pan_index%random.choice([5,6,8,10,12]) == 0:
                    pan1 += random.choice([-30, 30])
                axis = 'pan' # activates write to pan, with range control
                search_tilt_index = (search_tilt_index+1)%len(search_tiltpos)
                tilt1 = search_tiltpos[search_tilt_index]
                ser.write('t %03dn'%tilt1) # directly write tilt, since value is not random
            if axis == 'pan':
                # send pan position to head (eg. 'p 60')
                pan1 += int((value)*120)*gain(i)
                if pan1 < 5: 
                    pan1 += 180
                    memrec_turnoff = True
                if pan1 > 230: 
                    pan1 -= 180
                    memrec_turnoff = True
                if memrec_turnoff:
                    sender.send_json('memoryRecording 0')
                    time_reset_memrec = time.time()
                    memrec_turnon = True
                    memrec_turnoff = False
                    print 'ROBOCTRL TURN OFF MEMREC BEFORE HEAD SPIN'
                ser.write('p %03dn'%pan1)
            if axis == 'tilt':
                # send tilt position to head (eg. 'p 60')
                tilt1 += int((value)*90)*gain(i)
                if tilt1 > 40: tilt1 = 40-(tilt1-45)                
                if tilt1 < 2: tilt1 = 2-(tilt1-2)                
                ser.write('t %03dn'%tilt1)
            
        if robohead == 2:
            if axis == 'pan' and value == -1:
                seed = (random.random()-0.5)*2
                distance = 30            
                pan2 = int(pan2+(seed*distance))
                tilt2 = int(tilt2+((1-seed)*distance))
                if pan2 < 10: pan2 = 20
                if pan2 > 200: pan2 = 180
                if tilt2 < 20: tilt2 = 30
                if tilt2 > 200: tilt2 = 180
                ser.write('o %03dn'%pan2)
                ser.write('r %03dn'%tilt2)

        i += 1


if __name__ == '__main__':
    robocontrol('localhost')
