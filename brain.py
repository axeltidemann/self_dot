#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
from uuid import uuid1
from collections import deque
import sys
import glob
import cPickle as pickle
from subprocess import call
import time
import ctypes
import os

import numpy as np
import zmq
from sklearn import preprocessing as pp
from sklearn import svm
from scipy.io import wavfile
from scikits.samplerate import resample
import cv2
from sklearn.decomposition import RandomizedPCA
from scipy.signal.signaltools import correlate2d as c2d
import sai as pysai

import utils
import IO
import association
import my_sai_test as mysai

try:
    opencv_prefix = os.environ['VIRTUAL_ENV']
except:
    opencv_prefix = '/usr/local'
    print 'VIRTUAL_ENV variable not set, we are guessing OpenCV files reside in /usr/local - if OpenCV croaks, this is the reason.'
    
FACE_HAAR_CASCADE_PATH = opencv_prefix + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
EYE_HAAR_CASCADE_PATH = opencv_prefix + '/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
AUDIO_HAMMERTIME = 6 # Hamming distance match criterion
FACE_HAMMERTIME = 10
FRAME_SIZE = (160,120) # Neural network image size, 1/4 of full frame size.

# LOOK AT EYES? CAN YOU DETERMINE ANYTHING FROM THEM?
# PRESENT VISUAL INFORMATION - MOVE UP OR DOWN
def face_extraction(host, extended_search=False, show=False):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    eye_cascade = cv2.cv.Load(EYE_HAAR_CASCADE_PATH)
    face_cascade = cv2.cv.Load(FACE_HAAR_CASCADE_PATH)
    storage = cv2.cv.CreateMemStorage()

    context = zmq.Context()
    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, IO.CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    publisher = context.socket(zmq.PUB)
    publisher.bind('tcp://*:{}'.format(IO.FACE))

    robocontrol = context.socket(zmq.PUSH)
    robocontrol.connect('tcp://localhost:{}'.format(IO.ROBO))

    if show:
        cv2.namedWindow('Faces', cv2.WINDOW_NORMAL)
    i = 0
    j = 1
    while True:
        frame = utils.recv_array(camera).copy() # Weird, but necessary to do a copy.

        if j%2 == 0: # Every N'th frame we do face extraction.
            j = 1
        else:
            j += 1
            continue

        rows = frame.shape[1]
        cols = frame.shape[0]

        faces = [ (x,y,w,h) for (x,y,w,h),n in 
                  cv2.cv.HaarDetectObjects(cv2.cv.fromarray(frame), face_cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (50,50)) ] 

        if extended_search:
            eyes = [ (x,y,w,h) for (x,y,w,h),n in 
                     cv2.cv.HaarDetectObjects(cv2.cv.fromarray(frame), eye_cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (20,20)) ] 

            try: 
                if len(eyes) == 2:
                    x, y, _, _ = eyes[0]
                    x_, y_, _, _ = eyes[1]
                    angle = np.rad2deg(np.arctan( float((y_ - y))/(x_ - x) ))
                    rotation = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
                    frame = cv2.warpAffine(frame, rotation, (rows,cols))

                    faces.extend([ (x,y,w,h) for (x,y,w,h),n in 
                                   cv2.cv.HaarDetectObjects(cv2.cv.fromarray(frame), face_cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (50,50)) ])
            except Exception, e:
                print e, 'Eye detection failed.'

        # We select the biggest face.
        if faces:
            faces_sorted = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x,y,w,h = faces_sorted[0]
            x_diff = (x + w/2. - rows/2.)/rows
            y_diff = (y + h/2. - cols/2.)/cols
            utils.send_array(publisher, cv2.resize(frame[y:y+h, x:x+w], (100,100)))
            i += 1
            if i%5 == 0:
                if abs(x_diff) > .05:
                    robocontrol.send_json([ 1, 'pan', .52 if x_diff < 0 else .47]) 
                robocontrol.send_json([ 1, 'tilt', y_diff])
                i = 0
    
        if show:
            if faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
                cv2.line(frame, (x + w/2, y + h/2), (rows/2, cols/2), (255,0,0), 2)
                cv2.waitKey(1)
            cv2.imshow('Faces', frame)


def train_network(x, y, output_dim=100, leak_rate=.9, bias_scaling=.2, reset_states=True, use_pinv=True):
    import Oger
    import mdp

    mdp.numx.random.seed(7)

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=output_dim, 
                                              leak_rate=leak_rate, 
                                              bias_scaling=bias_scaling, 
                                              reset_states=reset_states)
    readout = mdp.nodes.LinearRegressionNode(use_pinv=use_pinv)
        
    net = mdp.hinet.FlowNode(reservoir + readout)
    net.train(x,y)

    return net

def dream(brain):
    return True # SELF-ORGANIZING OF ALL MEMORIES! EXTRACT CATEGORIES AND FEATURES!

            
def cochlear(filename, db=-40, stride=441, new_rate=22050, ears=1, apply_filter=1):
    rate, data = wavfile.read(filename)
    assert data.dtype == np.int16
    data = data / float(2**15)
    if rate != new_rate:
        data = resample(data, float(new_rate)/rate, 'sinc_best')
    data = data*10**(db/20)
    utils.array_to_csv('{}-audio.txt'.format(filename), data)
    call(['./carfac-cmd', filename, str(len(data)), str(ears), str(new_rate), str(stride), str(apply_filter)])
    naps = utils.csv_to_array('{}-output.txt'.format(filename))
    os.remove('{}-audio.txt'.format(filename))
    os.remove('{}-output.txt'.format(filename))
    return np.sqrt(np.maximum(0, naps)/np.max(naps))


def _predict_audio_id(audio_recognizer, NAP_resampled):
    x_test = audio_recognizer.rPCA.transform(np.ndarray.flatten(NAP_resampled))
    return audio_recognizer.predict(x_test)[0]


def _NAP_resampled(wav_file, maxlen, maxlen_scaled):
    NAP = utils.trim_right(utils.scale(utils.load_cochlear(wav_file)))
    return utils.zero_pad(resample(utils.trim_right(utils.scale(NAP)), float(maxlen)/NAP.shape[0], 'sinc_best'), maxlen_scaled), NAP

def calculate_sai_video_marginals(host, debug=False):
    me = mp.current_process()
    print '{} PID {}'.format(me.name, me.pid)

    context = zmq.Context()
    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    while True:
        pushbutton = eventQ.recv_json()
        if 'learn' in pushbutton:
            t0 = time.time()
            filename = pushbutton['filename']
            audio_segments = utils.get_segments(filename)

            NAP = cochlear(filename, stride=1, new_rate=22050, apply_filter=0)
            input_segment_width = 2048
            num_channels = NAP.shape[1]

            sai_params = mysai.CreateSAIParams(num_channels=num_channels,
                                               input_segment_width=input_segment_width,
                                               trigger_window_width=input_segment_width,
                                               sai_width=1024)
            sai = pysai.SAI(sai_params)


            input_segment_width = 2048
            norm_segments = np.rint(NAP.shape[0]*audio_segments/audio_segments[-1]).astype('int')

            for segment_id, NAP_segment in enumerate(utils.trim_right(NAP[norm_segments[i]:norm_segments[i+1]], threshold=.05) for i in range(len(norm_segments)-1)) :
                sai_video = [ np.copy(sai.RunSegment(input_segment.T)) for input_segment in utils.chunks(NAP_segment, input_segment_width) ]
                sai.Reset()
                marginals = [ mysai.sai_rectangles(frame) for frame in sai_video ]
                pickle.dump(marginals, open('{}-sai_video_marginals_segment_{}'.format(filename, segment_id), 'w'))

            print 'SAI video marginals for {} calculated in {} seconds'.format(filename, time.time() - t0)

def respond(control_host, learn_host, debug=False):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()
    
    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(control_host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    projector = context.socket(zmq.PUSH)
    projector.connect('tcp://{}:{}'.format(control_host, IO.PROJECTOR)) 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(control_host, IO.EXTERNAL))

    brainQ = context.socket(zmq.PULL)
    brainQ.bind('tcp://*:{}'.format(IO.BRAIN))
    
    association_in = context.socket(zmq.PUSH)
    association_in.connect('tcp://{}:{}'.format(learn_host, IO.ASSOCIATION_IN))

    association_out = context.socket(zmq.PULL)
    association_out.bind('tcp://*:{}'.format(IO.ASSOCIATION_OUT))
    
    poller = zmq.Poller()
    poller.register(eventQ, zmq.POLLIN)
    poller.register(brainQ, zmq.POLLIN)

    sound_to_face = []
    wordFace = {}
    face_to_sound = []
    faceWord = {}
    register = {}
    video_producer = {}
    voiceType1 = 0
    voiceType2 = 1
    wordSpace1 = 0.2
    wordSpaceDev1 = 0.3
    wordSpace2 = 0.2
    wordSpaceDev2 = 0.3
    
    if debug:
        import matplotlib.pyplot as plt
        plt.ion()
    
    while True:
        events = dict(poller.poll())
        if brainQ in events:
            cells = brainQ.recv_pyobj()

            mode = cells[0]
            wav_file = cells[1]

            if wav_file not in register:
                register[wav_file] = [False, False, False]
            
            if mode == 'audio_learn':
                register[wav_file][0] = cells

            if mode == 'video_learn':
                register[wav_file][1] = cells

            if mode == 'face_learn':
                register[wav_file][2] = cells

            if all(register[wav_file]):
                print 'Audio - video - face recognizers related to {} arrived at responder'.format(wav_file)
                # segment_ids: list of audio_ids in sentence
                _, _, segment_ids, wavs, wav_segments, audio_recognizer, maxlen, maxlen_scaled, NAP_hashes = register[wav_file][0]
                _, _, tarantino = register[wav_file][1]
                _, _, face_id, face_recognizer = register[wav_file][2]          

                for audio_id in segment_ids:
                    video_producer[(audio_id, face_id)] = tarantino
                    # By eliminating the last logical sentence, you can effectively get a statistical storage of audio_id.
                    if audio_id < len(sound_to_face) and not face_id in sound_to_face[audio_id]: # sound heard before, but not said by this face 
                        sound_to_face[audio_id].append(face_id)
                        #wordFace[audio_id].append([face_id,1])
                    else:
                        sound_to_face.append([face_id])
                    wordFace.setdefault(audio_id, [[face_id,0]])
                    found = 0
                    for item in wordFace[audio_id]:
                        if item[0] == face_id:
                            item[1] += 1
                            found = 1
                    if found == 0:
                        wordFace[audio_id].append([face_id,1])

                    # We can't go from a not known face to any of the sounds, that's just the way it is.
                    if face_id is not -1:
                        if face_id < len(face_to_sound) and not audio_id in face_to_sound[face_id]: #face seen before, but the sound is new
                            face_to_sound[face_id].append(audio_id)
                            #faceWord[face_id].append([audio_id,1])
                        else:
                            face_to_sound.append([audio_id])
                        faceWord.setdefault(face_id, [[audio_id,0]])
                        found = 0
                        for item in faceWord[face_id]:
                            if item[0] == audio_id:
                                item[1] += 1
                                found = 1
                        if found == 0:
                            faceWord[face_id].append([audio_id,1])

                del register[wav_file]
                
                similar_ids = []
                for audio_id in segment_ids:
                    new_audio_hash = NAP_hashes[audio_id][-1]
                    similar_ids_for_this_audio_id = [ utils.hamming_distance(new_audio_hash, np.random.choice(h)) for h in NAP_hashes ]
                    similar_ids.append(similar_ids_for_this_audio_id)
                #print '**wordFace', wordFace
                print '**faceWord', faceWord
                association_in.send_pyobj(['analyze',wav_file,wav_segments,segment_ids,wavs,similar_ids,wordFace,faceWord])
                
                                
        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'respond_single' in pushbutton:
                try:
                    filename = pushbutton['filename']
                    audio_segments = utils.get_segments(filename)
                    print 'Single response to {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    new_sentence = utils.load_cochlear(filename)
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    _,_,_,segmentData = utils.getSoundInfo(filename)
                    amps = [ item[0] for item in segmentData ]
                    segment_id = amps.index(max(amps))
                    #print 'Single selected to respond to segment {}'.format(segment_id)

                    NAP = utils.trim_right(utils.scale(new_sentence[norm_segments[segment_id]:norm_segments[segment_id+1]]))
                    NAP_resampled = utils.zero_pad(resample(NAP, float(maxlen)/NAP.shape[0], 'sinc_best'), maxlen_scaled)
                    
                    if debug:            
                        plt.imshow(NAP_resampled.T, aspect='auto')
                        plt.draw()
                    
                    try:
                        audio_id = _predict_audio_id(audio_recognizer, NAP_resampled)
                    except:
                        audio_id = 0
                        print 'Responding having only heard 1 sound.'

                    soundfile = np.random.choice(wavs[audio_id])

                    # segment start and end within sound file, if zero, play whole file
                    segstart, segend = wav_segments[(soundfile, audio_id)]

                    voiceChannel = 1
                    speed = 1
                    amp = -3 # voice amplitude in dB
                    #dur, maxamp = utils.getSoundParmFromFile(soundfile) # COORDINATION!
                    _,dur,maxamp,_ = utils.getSoundInfo(soundfile)
                    
                    start = 0
                    sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp))

                    print 'Recognized as sound {}'.format(audio_id)

                    face_id = np.random.choice(sound_to_face[audio_id])
                    video_time = (1000*dur)/IO.VIDEO_SAMPLE_TIME
                    stride = int(np.ceil(NAP.shape[0]/video_time))
                    projection = video_producer[(audio_id, face_id)](NAP[::stride])

                    for row in projection:
                        utils.send_array(projector, np.resize(row, FRAME_SIZE[::-1]))

                except:
                    utils.print_exception('Single response aborted.')

            if 'respond_sentence' in pushbutton:
                print 'SENTENCE Respond to', pushbutton['filename'][-12:]
                
                try:
                    filename = pushbutton['filename']
                    audio_segments = utils.get_segments(filename)
                    print 'Sentence response to {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    new_sentence = utils.load_cochlear(filename)
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    _,_,_,segmentData = utils.getSoundInfo(filename)
                    amps = [ item[0] for item in segmentData ]
                    segment_id = amps.index(max(amps))
                    print '**Sentence selected to respond to segment {}'.format(segment_id)

                    NAP = utils.trim_right(utils.scale(new_sentence[norm_segments[segment_id]:norm_segments[segment_id+1]]))
                    NAP_resampled = utils.zero_pad(resample(NAP, float(maxlen)/NAP.shape[0], 'sinc_best'), maxlen_scaled)
                    
                    if debug:            
                        plt.imshow(NAP_resampled.T, aspect='auto')
                        plt.draw()
                    
                    try:
                        audio_id = _predict_audio_id(audio_recognizer, NAP_resampled)
                    except:
                        audio_id = 0
                        print 'Responding having only heard 1 sound.'

                    numWords = len(segmentData)
                    print numWords
                    association_in.send_pyobj(['setParam', 'numWords', numWords ])
                    
                    association_in.send_pyobj(['makeSentence',audio_id])
                    print 'respond_sentence waiting for association output...'
                    sentence, secondaryStream = association_out.recv_pyobj()

                    print '*** Play sentence', sentence, secondaryStream
                    start = 0 
                    nextTime1 = 0
                    nextTime2 = 0
                    enableVoice2 = 1
                    for i in range(len(sentence)):
                        word_id = sentence[i]
                        soundfile = np.random.choice(wavs[word_id])
                        voiceChannel = 1
                        speed = 1
                        
                        # segment start and end within sound file, if zero, play whole file
                        segstart, segend = wav_segments[(soundfile, word_id)]
                        amp = -3 # voice amplitude in dB
                        #totaldur, maxamp = utils.getSoundParmFromFile(soundfile)
                        _,totaldur,maxamp,_ = utils.getSoundInfo(soundfile)
                        dur = segend-segstart
                        if dur <= 0: dur = totaldur
                        sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp))
                        #start += dur # if we want to create a 'score section' for Csound, update start time to make segments into a contiguous sentence
                        wordSpacing1 = wordSpace1 + np.random.random()*wordSpaceDev1
                        nextTime1 += (dur/speed)+wordSpacing1
                        #print 'voice 2 ready to play', secondaryStream[i], i
                        if enableVoice2:
                            word_id2 = secondaryStream[i]
                            #print 'voice 2 playing', secondaryStream[i]
                            soundfile2 = np.random.choice(wavs[word_id2])
                            voiceChannel2 = 2
                            start2 = 0.7 #  set delay between voice 1 and 2
                            speed2 = 0.7
                            amp2 = -10 # voice amplitude in dB
                            segstart2, segend2 = wav_segments[(soundfile2, word_id2)]
                            dur2 = segend2-segstart2
                            #totalDur2, maxamp2 = utils.getSoundParmFromFile(soundfile2)
                            _,totalDur2,maxamp2,_ = utils.getSoundInfo(soundfile)
                            if dur2 <= 0: dur2 = totalDur2
                            sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel2, voiceType2, start2, soundfile2, speed2, segstart2, segend2, amp2, maxamp2))
                            wordSpacing2 = wordSpace2 + np.random.random()*wordSpaceDev2
                            nextTime2 += (dur2/speed2)+wordSpacing2
                            #enableVoice2 = 0
                        # trig another word in voice 2 only if word 2 has finished playing (and sync to start of voice 1)
                        if nextTime1 > nextTime2: enableVoice2 = 1 

                        face_id = np.random.choice(sound_to_face[word_id])
                        video_time = (1000*dur)/IO.VIDEO_SAMPLE_TIME
                        stride = int(np.ceil(NAP.shape[0]/video_time))
                        projection = video_producer[(audio_id, face_id)](NAP[::stride])

                        for row in projection:
                            utils.send_array(projector, np.resize(row, FRAME_SIZE[::-1]))

                        # as the first crude method of assembling a sentence, just wait for the word duration here
                        time.sleep(dur+wordSpacing1)

                except:
                    utils.print_exception('Sentence response aborted.')
                    
            if 'testSentence' in pushbutton:
                print 'testSentence', pushbutton
                association_in.send_pyobj(['makeSentence',int(pushbutton['testSentence'])])
                print 'testSentence waiting for association output...'
                sentence, secondaryStream = association_out.recv_pyobj()
                print '*** Test sentence', sentence, secondaryStream
            
            if 'assoc_setParam' in pushbutton:
                parm, value = pushbutton['assoc_setParam'].split()
                association_in.send_pyobj(['setParam', parm, value ])

            if 'respond_setParam' in pushbutton:
                items = pushbutton['respond_setParam'].split()
                if items[0] == 'voiceType':
                    chan = items[1]
                    if chan == '1': voiceType1 = int(items[2])
                    if chan == '2': voiceType2 = int(items[2])
                if items[0] == 'wordSpace':
                    chan = items[1]
                    print 'wordSpace chan', chan, items
                    if chan == '1': wordSpace1 = float(items[2])
                    if chan == '2': wordSpace2 = float(items[2])
                if items[0] == 'wordSpaceDev':
                    chan = items[1]
                    print 'wordSpaceDev1 chan', chan, items
                    if chan == '1': wordSpaceDev1 = float(items[2])
                    if chan == '2': wordSpaceDev2 = float(items[2])

            if 'play_id' in pushbutton:
                try:
                    items = pushbutton['play_id'].split(' ')
                    if len(items) < 3: print 'PARAMETER ERROR: play_id audio_id voiceChannel voiceType'
                    play_audio_id = int(items[0])
                    voiceChannel = int(items[1])
                    voiceType = int(items[2])
                    print 'play_audio_id', play_audio_id, 'voice', voiceChannel
                    print 'wavs[play_audio_id]', wavs[play_audio_id]
                    #print wavs
                    soundfile = np.random.choice(wavs[play_audio_id])
                    
                    speed = 1
                    #print 'wav_segments', wav_segments
                    segstart, segend = wav_segments[(soundfile, play_audio_id)]
                    #segstart = 0 # segment start and end within sound file
                    #segend = 0 # if zero, play whole file
                    amp = -3 # voice amplitude in dB
                    #dur, maxamp = utils.getSoundParmFromFile(soundfile)
                    _,dur,maxamp,_ = utils.getSoundInfo(soundfile)
                    start = 0
                    sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType, start, soundfile, speed, segstart, segend, amp, maxamp))
                except:
                    utils.print_exception('play_id aborted.')

            if 'showme' in pushbutton:
                # just for inspecting the contents of objects while running 
                print 'printing '+pushbutton['showme']
                try:
                    o = compile('print '+pushbutton['showme'], '<string>','exec')
                    eval(o)
                except Exception, e:
                    print e, 'showme print failed.'

            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], me.name), [ sound_to_face, wordFace, face_to_sound, faceWord, video_producer, segment_ids, wavs, wav_segments, audio_recognizer, maxlen, maxlen_scaled, NAP_hashes, face_id, face_recognizer ])

            if 'load' in pushbutton:
                sound_to_face, wordFace, face_to_sound, faceWord, video_producer, segment_ids, wavs, wav_segments, audio_recognizer, maxlen, maxlen_scaled, NAP_hashes, face_id, face_recognizer = utils.load('{}.{}'.format(pushbutton['load'], me.name))

                    

def learn_audio(host, debug=False):
    me = mp.current_process()
    print '{} PID {}'.format(me.name, me.pid)

    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, IO.MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    stateQ, eventQ, brainQ = _three_amigos(context, host)
    
    poller = zmq.Poller()
    poller.register(mic, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    audio = deque()
    NAPs = []
    wavs = []
    wav_segments = {}
    NAP_hashes = []

    audio_recognizer = []
    maxlen = []
    maxlen_scaled = []

    state = stateQ.recv_json()
    
    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()

        if mic in events:
            new_audio = utils.recv_array(mic)
            if state['record']:
                audio.append(new_audio)

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton:
                try:
                    t0 = time.time()
                    filename = pushbutton['filename']
                    audio_segments = utils.get_segments(filename)

                    print 'Learning {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    new_sentence = utils.load_cochlear(filename)
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    segment_ids = []
                    for segment, new_sound in enumerate([ utils.trim_right(utils.scale(new_sentence[norm_segments[i]:norm_segments[i+1]])) for i in range(len(norm_segments)-1) ]):
                        # Do we know this sound?
                        if debug:
                            plt.imshow(new_sound.T, aspect='auto')
                            plt.draw()
                        
                        hammings = [ np.inf ]
                        new_audio_hash = utils.d_hash(new_sound)
                        audio_id = 0
                        if len(NAPs) == 1:
                            hammings = [ utils.hamming_distance(new_audio_hash, h) for h in NAP_hashes[0] ]
                        
                        if audio_recognizer:
                            resampled_new_sound = utils.zero_pad(resample(new_sound, float(maxlen)/new_sound.shape[0], 'sinc_best'), maxlen_scaled)
                            x_test = audio_recognizer.rPCA.transform(np.ndarray.flatten(resampled_new_sound))
                            audio_id = audio_recognizer.predict(x_test)[0]
                            hammings = [ utils.hamming_distance(new_audio_hash, h) for h in NAP_hashes[audio_id] ]

                        if np.mean(hammings) < AUDIO_HAMMERTIME:
                            NAPs[audio_id].append(new_sound)
                            NAP_hashes[audio_id].append(new_audio_hash)
                            wavs[audio_id].append(filename)
                            print 'Sound is similar to sound {}, hamming mean {}'.format(audio_id, np.mean(hammings))
                        else:
                            print 'New sound, hamming mean {} from sound {}'.format(np.mean(hammings), audio_id)
                            NAPs.append([new_sound])
                            NAP_hashes.append([new_audio_hash])
                            wavs.append([filename])
                            audio_id = len(NAPs) - 1

                        # The mapping from wavfile and audio ID to the segment within the audio file
                        wav_segments[(filename, audio_id)] = [ audio_segments[segment], audio_segments[segment+1] ]
                        segment_ids.append(audio_id)
                        
                        # Scale the sizes of the samples according to the biggest one. The idea is that this scale well. Otherwise, create overlapping bins.
                        maxlen = max([ m.shape[0] for memory in NAPs for m in memory ])
                        resampled_memories = [ [ resample(m, float(maxlen)/m.shape[0], 'sinc_best') for m in memory ] for memory in NAPs ]
                        maxlen_scaled = max([ m.shape[0] for memory in resampled_memories for m in memory ])

                        if len(NAPs) > 1:
                            resampled_memories = [ [ utils.zero_pad(m, maxlen_scaled) for m in memory ] for memory in resampled_memories ]
                            resampled_flattened_memories = [ np.ndarray.flatten(m) for memory in resampled_memories for m in memory ]
                            targets = [ i for i,f in enumerate(NAPs) for _ in f ]
                            
                            rPCA = RandomizedPCA(n_components=100)
                            x_train = rPCA.fit_transform(resampled_flattened_memories)

                            audio_recognizer = svm.LinearSVC()
                            audio_recognizer.fit(x_train, targets)
                            audio_recognizer.rPCA = rPCA

                    t1 = time.time()
                    brainQ.send_pyobj(['audio_learn', filename, segment_ids, wavs, wav_segments, audio_recognizer, maxlen, maxlen_scaled, NAP_hashes])
                    print 'Audio learned in {} seconds, ZMQ time {} seconds'.format(t1 - t0, time.time() - t1)
                except:
                    utils.print_exception('Audio learning aborted.')

                audio.clear()

            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], me.name), [ NAPs, wavs, wav_segments, NAP_hashes, audio_recognizer, maxlen, maxlen_scaled ])

            if 'load' in pushbutton:
                NAPs, wavs, wav_segments, NAP_hashes, audio_recognizer, maxlen, maxlen_scaled = utils.load('{}.{}'.format(pushbutton['load'], me.name))

                
def learn_video(host, debug=False):
    me = mp.current_process()
    print '{} PID {}'.format(me.name, me.pid)

    context = zmq.Context()

    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, IO.CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    stateQ, eventQ, brainQ = _three_amigos(context, host)
    
    poller = zmq.Poller()
    poller.register(camera, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    video = deque()

    state = stateQ.recv_json()

    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()
             
        if camera in events:
            new_video = utils.recv_array(camera)
            if state['record']:
                frame = cv2.resize(new_video, FRAME_SIZE)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.
                gray_flattened = np.ndarray.flatten(gray_image)
                video.append(gray_flattened)

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton:
                try:
                    t0 = time.time()
                    filename = pushbutton['filename']
                    new_sentence = utils.trim_right(utils.scale(utils.load_cochlear(filename)))
                    
                    video_segment = np.array(list(video))
                    if video_segment.shape[0] == 0:
                        video_segment = np.array([ np.ndarray.flatten(np.zeros(FRAME_SIZE)) for _ in range(10) ])
                        print 'No video recorded. Using black image as stand-in.'

                    NAP_len = new_sentence.shape[0]
                    video_len = video_segment.shape[0]
                    stride = int(max(1,np.floor(float(NAP_len)/video_len)))
                    
                    x = new_sentence[:NAP_len - np.mod(NAP_len, stride*video_len):stride]
                    if x.shape[0] == 0:
                        x = new_sentence[::stride]
                    y = video_segment[:x.shape[0]]

                    tarantino = train_network(x,y, output_dim=10)
                    tarantino.stride = stride

                    t1 = time.time()
                    brainQ.send_pyobj([ 'video_learn', filename, tarantino ])
                    print 'Video learned in {} seconds, ZMQ time {} seconds'.format(t1 - t0, time.time() - t1)
                except:
                    utils.print_exception('Video learning aborted.')

                video.clear()

                
def learn_faces(host, debug=False):
    me = mp.current_process()
    print '{} PID {}'.format(me.name, me.pid)

    context = zmq.Context()

    face = context.socket(zmq.SUB)
    face.connect('tcp://{}:{}'.format(host, IO.FACE))
    face.setsockopt(zmq.SUBSCRIBE, b'')

    stateQ, eventQ, brainQ = _three_amigos(context, host)
    
    poller = zmq.Poller()
    poller.register(face, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    faces = deque()
    face_history = []
    face_hashes = []
    face_recognizer = []

    state = stateQ.recv_json()
    
    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()
        
        if face in events:
            new_face = utils.recv_array(face)
            gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY) / 255.

            if state['record']:
                faces.append(gray)

            # TODO: move to respond
            # if state['facerecognition']:
            #     try: 
            #         face_id = face_recognizer.predict(np.ndarray.flatten(gray))[0]
            #         print 'Face {} has previously said {}'.format(face_id, face_to_sound[face_id])
                                        
            #     except Exception, e:
            #         print e, 'Face recognition aborted.'

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton:
                try:
                    t0 = time.time()
                    filename = pushbutton['filename']

                    # Do we know this face?
                    hammings = [ np.inf ]
                    new_faces = list(faces)
                    new_faces_hashes = [ utils.d_hash(f) for f in new_faces ]

                    face_id = -1
                    if new_faces:
                        face_id = 0
                        if len(face_history) == 1:
                            hammings = [ utils.hamming_distance(f, m) for f in new_faces_hashes for m in face_hashes[0] ]

                        if face_recognizer:
                            x_test = [ face_recognizer.rPCA.transform(np.ndarray.flatten(f)) for f in new_faces ]
                            predicted_faces = [ face_recognizer.predict(x)[0] for x in x_test ]
                            uniq = np.unique(predicted_faces)
                            face_id = uniq[np.argsort([ sum(predicted_faces == u) for u in uniq ])[-1]]
                            hammings = [ utils.hamming_distance(f, m) for f in new_faces_hashes for m in face_hashes[face_id] ]

                        if np.mean(hammings) < FACE_HAMMERTIME:
                            face_history[face_id].extend(new_faces)
                            face_hashes[face_id].extend(new_faces_hashes)
                            print 'Face is similar to face {}, hamming mean {}'.format(face_id, np.mean(hammings))
                        else:
                            print 'New face, hamming mean {} from face {}'.format(np.mean(hammings), face_id)
                            face_history.append(new_faces)
                            face_hashes.append(new_faces_hashes)
                            face_id = len(face_history) - 1

                        if len(face_history) > 1:
                            # Possible fast version: train only on last face.

                            x_train = [ np.ndarray.flatten(f) for cluster in face_history for f in cluster ]
                            targets = [ i for i,f in enumerate(face_history) for _ in f ]

                            rPCA = RandomizedPCA(n_components=100)
                            x_train = rPCA.fit_transform(x_train)
                            face_recognizer = svm.LinearSVC()
                            face_recognizer.fit(x_train, targets)
                            face_recognizer.rPCA = rPCA
                    else:
                        print 'Face not detected.'

                    t1 = time.time()
                    brainQ.send_pyobj([ 'face_learn', filename, face_id, face_recognizer ])
                    print 'Faces learned in {} seconds, ZMQ time {} seconds'.format(t1 - t0, time.time() - t1)
                except:
                    utils.print_exception('Face learning aborted.')

                faces.clear()

            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], me.name), [ face_history, face_hashes, face_recognizer ])

            if 'load' in pushbutton:
                face_history, face_hashes, face_recognizer = utils.load('{}.{}'.format(pushbutton['load'], me.name))


def _three_amigos(context, host):
    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, IO.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    brainQ = context.socket(zmq.PUSH)
    brainQ.connect('tcp://{}:{}'.format(host, IO.BRAIN))

    return stateQ, eventQ, brainQ
