#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import os
import random
import utils

import cv2
import numpy as np
import zmq

from utils import send_array, recv_array
import myCsoundAudioOptions

# Milliseconds between each image capture
VIDEO_SAMPLE_TIME = 100
FRAME_SIZE = (640,480)

# ØMQ ports
CAMERA = 5561
PROJECTOR = 5562
MIC = 5563
SPEAKER = 5564
STATE = 5565
EXTERNAL = 5566 # If you change this, you're out of luck.
SNAPSHOT= 5567
EVENT = 5568
SCHEDULER = 5569
ROBO = 5570
#ROBOBACK = 5571
FACE = 5572
BRAIN = 5573
ASSOCIATION = 5574 

def video():
    me = mp.current_process()
    print me.name, 'PID', me.pid

    cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
    camera = cv2.VideoCapture(0)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind('tcp://*:{}'.format(CAMERA))

    projector = context.socket(zmq.PULL)
    projector.bind('tcp://*:{}'.format(PROJECTOR))
    
    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://localhost:{}'.format(STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'')
    
    poller = zmq.Poller()
    poller.register(stateQ, zmq.POLLIN)
    poller.register(projector, zmq.POLLIN)

    state = stateQ.recv_json()
    
    while True:
        events = dict(poller.poll(timeout=0))

        # This probably should be a pushbutton event instead - now this will be done every time the state is updated.
        if stateQ in events:
            state = stateQ.recv_json()        
            if state['fullscreen'] > 0:
                cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
            if state['display2'] > 0:
                cv2.moveWindow('Output', 2000, 100)

        if projector in events:
            cv2.imshow('Output', cv2.resize(recv_array(projector), FRAME_SIZE))
        else:
            cv2.imshow('Output', np.zeros(FRAME_SIZE[::-1]))
        
        _, frame = camera.read()
        frame = cv2.resize(frame, FRAME_SIZE)
        send_array(publisher, frame)

        cv2.waitKey(VIDEO_SAMPLE_TIME)

def audio():
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind('tcp://*:{}'.format(MIC))

    robocontrol = context.socket(zmq.PUSH)
    robocontrol.connect('tcp://localhost:{}'.format(ROBO))

    subscriber = context.socket(zmq.PULL)
    subscriber.bind('tcp://*:{}'.format(SPEAKER))

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://localhost:{}'.format(STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://localhost:{}'.format(EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://localhost:{}'.format(EXTERNAL))

    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    import time
    memRecPath = myCsoundAudioOptions.memRecPath
        
    import csnd6
    cs = csnd6.Csound()
    arguments = csnd6.CsoundArgVList()
    arguments.Append("dummy")
    arguments.Append("self_dot.csd")
    csoundCommandline = myCsoundAudioOptions.myAudioDevices
    comlineParmsList = csoundCommandline.split(' ')
    for item in comlineParmsList:
        arguments.Append("%s"%item)
    cs.Compile(arguments.argc(), arguments.argv())
    stopflag = 0
    zeroChannelsOnNoBrain = 1
    
    # optimizations to avoid function lookup inside loop
    tGet = cs.TableGet 
    tSet = cs.TableSet
    cGet = cs.GetChannel
    cSet = cs.SetChannel
    perfKsmps = cs.PerformKsmps

    filename = []
    counter = 0
    ampPitchCentroid = [[],[],[]]
    
    ambientFiles = [] # used by the ambient sound generator (instr 90 pp)
    ambientActive = 0

    state = stateQ.recv_json()
    
    while not stopflag:
        counter += 1
        counter = counter%16000 # just to reset sometimes
        stopflag = cs.PerformKsmps()

        events = dict(poller.poll(timeout=0))

        if stateQ in events:
            state = stateQ.recv_json()

        # get Csound channel data
        audioStatus = cGet("audioStatus")           
        audioStatusTrig = cGet("audioStatusTrig")       # signals start of a statement (audio in)
        transient = cGet("transient")                   # signals start of a segment within a statement (audio in)        
        memRecTimeMarker = cGet("memRecTimeMarker")     # (in memRec) get the time since start of statement
        memRecActive = cGet("memRecActive")             # flag to check if memoryRecording is currently recording to file in Csound
        memRecMaxAmp = cGet("memRecMaxAmp")             # max amplitude for each recorded file
        panposition = cs.GetChannel("panalyzer_pan")
        in_amp = cs.GetChannel("in_amp")
        in_pitch = cs.GetChannel("in_pitch")
        in_centroid = cs.GetChannel("in_centroid")

        if state['roboActive'] > 0:
            if (panposition < 0.48) or (panposition > 0.52):
                print 'panposition', panposition
                robocontrol.send_json([1,'pan',panposition-.5])
            if (counter % 500) == 0:
                robocontrol.send_json([2,'pan',-1])
         
        if state['ambientSound'] > 0:
            if ambientActive == 0:
                cs.InputMessage('i 92 0 -1')
                ambientActive = 1
            if (counter % 4000) == 0:
                #print 'Old ambient files:', ambientFiles
                #print 'update ambient sound ftables.'
                newtable, ambientFiles = utils.updateAmbientMemoryWavs(ambientFiles)
                #print 'newtable, ambientFiles', newtable, ambientFiles
                cs.InputMessage('i 90 0 4 "%s"'%newtable)
        
        if state['ambientSound'] == 0:
            if ambientActive == 1:
                cs.InputMessage('i -92 0 1')
                ambientActive = 0

        if state['memoryRecording']:
            if audioStatusTrig > 0:
                timestr = time.strftime('%Y_%m_%d_%H_%M_%S')
                print 'starting memoryRec', timestr
                tim_time = time.time()
                filename = memRecPath+timestr+'.wav'
                cs.InputMessage('i 34 0 -1 "%s"'%filename)
                markerfileName = memRecPath+timestr+'.txt'
                markerfile = open(markerfileName, 'w')
                markerfile.write('Self. audio clip perceived at %s\n'%tim_time)
                segmentstring = 'Sub segments (start, amp, pitch, cent): \n'
                segStart = 0.0
                ampPitchCentroid = [[],[],[]]
            if audioStatus > 0:
                ampPitchCentroid[0].append(in_amp)
                ampPitchCentroid[1].append(in_pitch)
                ampPitchCentroid[2].append(in_centroid)
            if (transient > 0) & (memRecActive > 0):
                if memRecTimeMarker == 0: pass
                else:
                    print '... ...get medians and update segments'
                    ampPitchCentroid[0].sort()
                    l = ampPitchCentroid[0]
                    ampMean = max(l)#np.mean(l[int(len(l)*0.5):int(len(l)*1)])
                    ampPitchCentroid[1].sort()
                    l = ampPitchCentroid[1]
                    pitchMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.75)])
                    ampPitchCentroid[2].sort()
                    l = ampPitchCentroid[2]
                    centroidMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.9)])
                    ampPitchCentroid = [[],[],[]]
                    segmentstring += '%.3f %.3f %.3f %.3f\n'%(segStart,ampMean,pitchMean,centroidMean)
                    segStart = memRecTimeMarker 
            if (audioStatusTrig < 0) & (memRecActive > 0):
                print '... ...get final medians and update segments'
                ampPitchCentroid[0].sort()
                l = ampPitchCentroid[0]
                ampMean = max(l)#np.mean(l[int(len(l)*0.5):int(len(l)*1)])
                ampPitchCentroid[1].sort()
                l = ampPitchCentroid[1]
                pitchMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.75)])
                ampPitchCentroid[2].sort()
                l = ampPitchCentroid[2]
                centroidMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.9)])
                ampPitchCentroid = [[],[],[]]
                segmentstring += '%.3f %.3f %.3f %.3f\n'%(segStart,ampMean,pitchMean,centroidMean)
                cs.InputMessage('i -34 0 1')
                markerfile.write(segmentstring)
                markerfile.write('Total duration: %f\n'%memRecTimeMarker)
                markerfile.write('\nMax amp for file: %f'%memRecMaxAmp)
                markerfile.close()
                print 'stopping memoryRec'
                #assoc.send_json(markerfileName)

        if not state['memoryRecording'] and memRecActive:
            print '... ...turnoff rec, get final medians and update segments'
            ampPitchCentroid[0].sort()
            l = ampPitchCentroid[0]
            ampMean = max(l)#np.mean(l[int(len(l)*0.5):int(len(l)*1)])
            ampPitchCentroid[1].sort()
            l = ampPitchCentroid[1]
            pitchMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.75)])
            ampPitchCentroid[2].sort()
            l = ampPitchCentroid[2]
            centroidMean = np.mean(l[int(len(l)*0.25):int(len(l)*0.9)])
            ampPitchCentroid = [[],[],[]]
            segmentstring += '%.3f %.3f %.3f %.3f\n'%(segStart,ampMean,pitchMean,centroidMean)
            cs.InputMessage('i -34 0 1')
            markerfile.write(segmentstring)
            markerfile.write('Total duration: %f'%memRecTimeMarker)
            markerfile.close()
            print 'stopping memoryRec'
            #assoc.send_json(markerfileName)

        interaction = []
                                                    
        if state['autolearn'] or state['autorespond_single'] or state['autorespond_sentence']:
            if audioStatusTrig > 0:
                sender.send_json('startrec')
                sender.send_json('_audioLearningStatus 1')
            if audioStatusTrig < 0:
                sender.send_json('stoprec')
                sender.send_json('_audioLearningStatus 0')
                if filename:
                    if state['autolearn']:
                        interaction.append('learnwav {}'.format(os.path.abspath(filename)))
                    if state['autorespond_single']:
                        interaction.append('respondwav_single {}'.format(os.path.abspath(filename)))
                    if state['autorespond_sentence']:
                        interaction.append('respondwav_sentence {}'.format(os.path.abspath(filename)))

        if interaction:
            sender.send_json('calculate_cochlear {}'.format(os.path.abspath(filename)))

            for command in interaction:
                sender.send_json(command)

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'selfvoice' in pushbutton:
                    print 'not implemented'

            if 'inputLevel' in pushbutton:
                mode = pushbutton['inputLevel']
                if mode == 'mute':
                    cs.InputMessage('i 21 0 .1 0')
                    print 'Mute'
                if mode == 'unmute':
                    cs.InputMessage('i 21 0 .1 1')
                    print 'Un-mute'
                if mode == 'reset': 
                    cs.InputMessage('i 21 0 .1 0')
                    cs.InputMessage('i 21 1 .1 1')

            if 'calibrateAudio' in pushbutton:
                cs.InputMessage('i -17 0 1') # turn off old noise gate
                cs.InputMessage('i 12 0 4') # measure roundtrip latency
                cs.InputMessage('i 13 4 1.9') # get audio input noise print
                cs.InputMessage('i 14 6 -1 1.0 1.0') # enable noiseprint and self-output suppression
                cs.InputMessage('i 15 6.2 2') # get noise floor level 
                cs.InputMessage('i 16 8.3 0.1') # set noise gate shape
                cs.InputMessage('i 17 8.5 -1') # turn on new noise gate

            if 'csinstr' in pushbutton:
                # generic csound instr message
                cs.InputMessage('{}'.format(pushbutton['csinstr']))
                print 'sent {}'.format(pushbutton['csinstr'])

            if 'selfDucking' in pushbutton:
                value = pushbutton['selfDucking']
                cs.InputMessage('i 22 0 1 "selfDucking" %f'%float(value))

            if 'zerochannels' in pushbutton:
                zeroChannelsOnNoBrain = int('{}'.format(pushbutton['zerochannels']))

            if 'playfile' in pushbutton:
                #print '[self.] playfile {}'.format(pushbutton['playfile'])
                try:
                    params = pushbutton['playfile']
                    voiceChannel, voiceType, start, soundfile, speed, segstart, segend, amp, maxamp = params.split(' ')
                    soundfile = str(soundfile)
                    voiceChannel = int(voiceChannel) # internal or external voice (primary/secondary associations)
                    instr = 60 + int(voiceType)
                    start = float(start)
                    segstart = float(segstart)
                    segend = float(segend)
                    amp = float(amp)
                    maxamp = float(maxamp)
                    speed = float(speed)
                    if voiceChannel == 2:
                        delaySend = -26 # delay send in dB
                        reverbSend = -23 # reverb send in dB
                    else:
                        delaySend = -96
                        reverbSend = -96 
                    csMessage = 'i %i %f 1 "%s" %f %f %f %f %i %f %f %f' %(instr, start, soundfile, segstart, segend, amp, maxamp, voiceChannel, delaySend, reverbSend, speed)
                    #print 'csMessage', csMessage                 
                    cs.InputMessage(csMessage)

                except Exception, e:
                    print e, 'Playfile aborted.'

