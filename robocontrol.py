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
if serialAvailable:
    while not connected:
        serin = ser.read()
        connected = True
        print 'Robot connected'
        
    time.sleep(2)
    print 'set initial position'
    # set tilt
    ser.write('t 045n')
    ser.write('p 000n')
    print 't'
    print 'initial position is set'
    time.sleep(1)

def robocontrol(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    robo = context.socket(zmq.SUB)
    robo.connect('tcp://{}:{}'.format(host, IO.ROBO))
    robo.setsockopt(zmq.SUBSCRIBE, b'')

    roboback = context.socket(zmq.PUB)
    roboback.bind('tcp://*:{}'.format(IO.ROBOBACK))

    #timeStamp = time.time()
    pos = 45
    while True:
    	#print 'robocontrol is running %i', time.time()-timeStamp
    	time.sleep(.05)
        message = None
        try:
            message = robo.recv_json(flags=zmq.DONTWAIT)
        except:
            message = 'hellllo'
        print 'message', message
        panposition = robo.recv_json()
        #print 'panposition', panposition
        # send pan position to head (eg. 'p 60')
        pos += int((panposition-0.5)*80)
        if pos < 10: pos += 180
        if pos > 200: pos -= 180
        command = 'p %03dn'%pos
        ser.write(command)
        #serRead = ser.readline()
        #print 'read * ', serRead, '*'
        #cs.SetChannel("panGate", 0)
        roboback.send_json(0)



if __name__ == '__main__':
    robocontrol('localhost')
