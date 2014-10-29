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
pan1 = 70
tilt1 = 45
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
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    robo = context.socket(zmq.PULL)
    robo.bind('tcp://*:{}'.format(IO.ROBO))

    #roboback = context.socket(zmq.PUB)
    #roboback.bind('tcp://*:{}'.format(IO.ROBOBACK))

    timeStamp = time.time()
    global pan1, tilt1, pan2, tilt2
    #print 'robocontrol entering loop %i', time.time()-timeStamp
    while True:
    	#print 'robocontrol is running %i', time.time()-timeStamp
    	time.sleep(.05)
        robohead,axis,value = robo.recv_json()
        if robohead == 1:
            if axis == 'pan':
                print 'head 1 panposition', value
                # send pan position to head (eg. 'p 60')
                pan1 += int((value-0.5)*80)
                #pan1 += int((value)*80)
                if pan1 < 10: pan1 += 180
                if pan1 > 200: pan1 -= 180
                ser.write('p %03dn'%pan1)
            if axis == 'tilt':
                print 'head 1 tiltposition', value
                # send til position to head (eg. 'p 60')
                #tilt1 += int((value-0.5)*3)
                tilt1 += int((value)*10)
                tilt1 = np.clip(tilt1, 10, 90)
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

            


if __name__ == '__main__':
    robocontrol('localhost')
