#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import zmq
import IO
import time
import random
from collections import namedtuple
import cPickle as pickle

# connect to arduino
import serial
import numpy as np

import utils
import zmq_ports

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
    robo.bind('tcp://*:{}'.format(zmq_ports.ROBO))

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, zmq_ports.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    poller = zmq.Poller()
    poller.register(robo, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, zmq_ports.EXTERNAL))

    timeStamp = time.time()
    global pan1, tilt1, pan2, tilt2
    time_reset_memrec = time.time()
    memrec_turnon = False
    memrec_turnoff = False
    search_pan_index = 0
    search_tilt_index = 0
    search_tiltpos = [20, 10, 30, 20, 5, 15, 25, 35, 20, 25, 15, 5, 0, 10, 20, 30, 40, 30]

    exp_decay = lambda g, a, x: g*np.exp(-a*x)

    i = 1000
    pan1_k = 0
    tilt1_k = 0

    pan_direction = -1

    stop = lambda x: 0
    pan1_movement_func = stop
    tilt1_movement_func = stop

    def pan_seq(x):
        x = np.mod(x,20)
        if x < 7:
            return 1
        if x < 14:
            return 0
        return -5

    def tilt_seq(x):
        x = np.mod(x,2)
        if x < 1:
            return 5
        if x < 2:
            return -5

    movement_functions = [ [lambda x: exp_decay(pan_direction*30, 10, x), stop], # left - right
                           [lambda x: exp_decay(pan_direction*30, 10, x), lambda x: 10*np.sin(-2*np.pi*x/.8) if x < .8 else 0 ], # headbanging
                           [pan_seq, tilt_seq] ] # stupid, but demo of sequences

    transient_timer = time.time()
    state = stateQ.recv_json()
    transient_face_mask_time = 3

    motor_cmd = utils.MotorCommand(1, 'dummy', 0, 0)

    while True:
    	time.sleep(.05)
        events = dict(poller.poll(timeout=100))

        if memrec_turnon and (time.time()-time_reset_memrec > 1.7):
            sender.send_json('memoryRecording 1')
            memrec_turnon = False
            print 'ROBOCTRL TURN ON MEMREC AFTER HEAD SPIN'

        if stateQ in events:
            state = stateQ.recv_json()

        if robo in events:
            motor_cmd = robo.recv_pyobj()
            if motor_cmd.robohead == 1:
                if state['musicMode']:
                    if motor_cmd.mode == 'transient':
                        i = 0
                        print '\t|\\-. Transient'
                        pan_direction = -pan_direction
                        pan1_movement_func, tilt1_movement_func = movement_functions[0]
                        # if time.time() - transient_timer > 10:
                        #     pan1_movement_func, tilt1_movement_func = random.choice(movement_functions)
                        #     transient_timer = time.time()
                else:
                    if motor_cmd.mode == 'transient': 
                        i = 0
                        pan1_movement_func = lambda x: exp_decay(10*motor_cmd.x_diff, 2, x)
                        tilt1_movement_func = stop
                        transient_timer = time.time()
                        print '\t|\\-. Transient, face search suppressed for {} seconds'.format(transient_face_mask_time)

                if motor_cmd.mode == 'face' and time.time() - transient_timer > transient_face_mask_time:
                    i = 0
                    pan1_movement_func = lambda x: exp_decay(motor_cmd.x_diff, 2, x)
                    tilt1_movement_func = lambda x: exp_decay(motor_cmd.y_diff, 2, x)

        # robohead 1 movement
        # if mode == 'pan': #when we have adjustment due to sound input...
        #     search_pan_index = 0 #do only fine adjustment
        if motor_cmd.mode == 'search' and time.time() - transient_timer > 2: #Face search
            
            # Also ugly - search parameters should be determined where they are sent from.
            pan1_movement_func = stop
            tilt1_movement_func = stop

            pan1 += (random.random()-0.5)*20
            search_pan_index += 1
            search_pan_index %= 1000
            if search_pan_index%random.choice([5,6,8,10,12]) == 0:
                pan1 += random.choice([-30, 30])
            search_tilt_index = (search_tilt_index+1)%len(search_tiltpos)
            tilt1 = search_tiltpos[search_tilt_index]
            ser.write('t %03dn'%tilt1) # directly write tilt, since value is not random
        #if mode == 'pan':
        # send pan position to head (eg. 'p 60')
        pan1 += pan1_movement_func(i)
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
        #if mode == 'tilt':
        # send tilt position to head (eg. 't 60')
        tilt1 += tilt1_movement_func(i)
        if tilt1 > 40: tilt1 = 40-(tilt1-45)                
        if tilt1 < 2: tilt1 = 2-(tilt1-2)                
        ser.write('t %03dn'%tilt1)

        # if robohead == 2:
        #     if mode == 'pan' and value == -1:
        #         seed = (random.random()-0.5)*2
        #         distance = 30            
        #         pan2 = int(pan2+(seed*distance))
        #         tilt2 = int(tilt2+((1-seed)*distance))
        #         if pan2 < 10: pan2 = 20
        #         if pan2 > 200: pan2 = 180
        #         if tilt2 < 20: tilt2 = 30
        #         if tilt2 > 200: tilt2 = 180
        #         ser.write('o %03dn'%pan2)
        #         ser.write('r %03dn'%tilt2)

        i += .1


if __name__ == '__main__':
    robocontrol('localhost')
