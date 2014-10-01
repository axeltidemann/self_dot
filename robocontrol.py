#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import zmq
import IO
import time
import random

# connect to arduino
import serial

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
pan1 = 0
tilt1 = 55
pan2 = 60
tilt2 = 45
if serialAvailable:
    while not connected:
        serin = ser.read()
        connected = True
        print 'Robot connected'
        
    time.sleep(2)
    print 'set initial position'
    ser.write('t %03dn'%tilt1)
    ser.write('p %03dn'%pan1)
    time.sleep(0.3)
    ser.write('r %03dn'%tilt1)
    ser.write('o %03dn'%pan1)
    print '...' 
    print 'initial position is set'
    time.sleep(1)

def robocontrol(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    robo = context.socket(zmq.SUB)
    robo.connect('tcp://{}:{}'.format(host, IO.ROBO))
    robo.setsockopt(zmq.SUBSCRIBE, b'')

    #roboback = context.socket(zmq.PUB)
    #roboback.bind('tcp://*:{}'.format(IO.ROBOBACK))

    #timeStamp = time.time()
    global pan1, tilt1, pan2, tilt2
    while True:
    	#print 'robocontrol is running %i', time.time()-timeStamp
    	time.sleep(.05)
        robohead,axis,value = robo.recv_json()
        if robohead == 1:
            if axis == 'pan':
                print 'head 1 panposition', value
                # send pan position to head (eg. 'p 60')
                pan1 += int((value-0.5)*80)
                if pan1 < 10: pan1 += 180
                if pan1 > 200: pan1 -= 180
                ser.write('p %03dn'%pan1)



if __name__ == '__main__':
    robocontrol('localhost')
