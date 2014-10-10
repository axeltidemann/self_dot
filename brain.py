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

import utils
import IO
import analyze_associations2 as association

try:
    opencv_prefix = os.environ['VIRTUAL_ENV']
except:
    opencv_prefix = '/usr/local'
    print 'VIRTUAL_ENV variable not set, we are guessing OpenCV files reside in /usr/local - if OpenCV croaks, this is the reason.'
    
FACE_HAAR_CASCADE_PATH = opencv_prefix + '/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
EYE_HAAR_CASCADE_PATH = opencv_prefix + '/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
HAMMERTIME = 10 # Hamming distance match criterion
FRAME_SIZE = (160, 90) # Neural network image size

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

    if show:
        cv2.namedWindow('Faces', cv2.WINDOW_NORMAL)

    while True:
        frame = utils.recv_array(camera).copy() # Weird, but necessary to do a copy.

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
            utils.send_array(publisher, cv2.resize(frame[y:y+h, x:x+w], (100,100)))
    
        if show:
            if faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)
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

            
def cochlear(filename, db=-40, stride=441, threshold=.025, new_rate=22050, ears=1, channels=71):
    rate, data = wavfile.read(utils.wait_for_wav(filename))
    assert data.dtype == np.int16
    data = data / float(2**15)
    data = resample(data, float(new_rate)/rate, 'sinc_best')
    data = data*10**(db/20)
    utils.array_to_csv('{}-audio.txt'.format(filename), data)
    call(['./carfac-cmd', filename, str(len(data)), str(ears), str(channels), str(new_rate), str(stride)])
    naps = utils.csv_to_array('{}-output.txt'.format(filename))
    return np.sqrt(np.maximum(0, naps)/np.max(naps))


def respond(control_host, learn_host):
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

    brainQ = context.socket(zmq.SUB)
    brainQ.connect('tcp://{}:{}'.format(learn_host, IO.BRAIN))
    brainQ.setsockopt(zmq.SUBSCRIBE, b'') 
        
    poller = zmq.Poller()
    poller.register(eventQ, zmq.POLLIN)
    poller.register(brainQ, zmq.POLLIN)
        
    while True:
        events = dict(poller.poll())
        if brainQ in events:
            wavs, wav_segments, sound_to_face, face_to_sound, audio_recognizer, video_producer, maxlen,  maxlen_scaled = brainQ.recv_pyobj()

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'respond_single' in pushbutton:
                print 'Respond to', pushbutton['filename']

                try:
                    NAP = cochlear(pushbutton['filename'])
                    NAP_resampled = utils.zero_pad(resample(utils.trim_right(utils.scale(NAP)), float(maxlen)/NAP.shape[0], 'sinc_best'), maxlen_scaled)

                    try:
                        audio_id = audio_recognizer.predict(np.ndarray.flatten(NAP_resampled))[0]
                    except:
                        audio_id = 0
                        print 'Responding having only heard 1 sound.'

                    soundfile = np.random.choice(wavs[audio_id])

                    # segment start and end within sound file, if zero, play whole file
                    segstart, segend = wav_segments[(soundfile, audio_id)]

                    voiceChannel = 1
                    voiceType = 1 
                    speed = 1
                    amp = -3 # voice amplitude in dB
                    dur, maxamp = utils.getSoundParmFromFile(soundfile)
                    start = 0
                    sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType, start, soundfile, speed, segstart, segend, amp, maxamp))

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
                    NAP = cochlear(pushbutton['filename'])

                    NAP_resampled = utils.zero_pad(resample(utils.trim_right(utils.scale(NAP)), float(maxlen)/NAP.shape[0], 'sinc_best'), maxlen_scaled)

                    try:
                        audio_id = audio_recognizer.predict(np.ndarray.flatten(NAP_resampled))[0]
                    except:
                        audio_id = 0
                        print 'Responding having only heard 1 sound.'

                    numWords = 4
                    method = 'boundedAdd'
                    timeBeforeWeight = 0.0
                    timeAfterWeight = 0.5
                    timeDistance = 5.0
                    durationWeight = 0.1
                    posInSentenceWeight = 0.5
                    method2 = 'boundedAdd'
                    timeBeforeWeight2 = 0.5
                    timeAfterWeight2 = 0.0
                    timeDistance2 = 5.0
                    durationWeight2 = 0.5
                    posInSentenceWeight2 = 0.5                                                        
                    sentence, secondaryStream = association.makeSentence(audio_id, numWords, 
                                                        method, timeBeforeWeight, timeAfterWeight, timeDistance, durationWeight, posInSentenceWeight,
                                                        method2, timeBeforeWeight2, timeAfterWeight2, timeDistance2, durationWeight2, posInSentenceWeight2)

                    print '*** Play sentence', sentence, secondaryStream
                    start = 0 
                    nextTime1 = 0
                    nextTime2 = 0
                    enableVoice2 = 1
                    for i in range(len(sentence)):
                        word_id = sentence[i]
                        soundfile = np.random.choice(wavs[word_id])
                        voiceChannel = 1
                        voiceType = 0
                        speed = 1
                        
                        # segment start and end within sound file, if zero, play whole file
                        segstart, segend = wav_segments[(soundfile, word_id)]
                        amp = -3 # voice amplitude in dB
                        dur, maxamp = utils.getSoundParmFromFile(soundfile)
                        sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType, start, soundfile, speed, segstart, segend, amp, maxamp))
                        #start += dur # if we want to create a 'score section' for Csound, update start time to make segments into a contiguous sentence
                        nextTime1 += (dur/speed)

                        if enableVoice2:
                            word_id2 = secondaryStream[i]
                            soundfile2 = np.random.choice(wavs[word_id2])
                            voiceChannel2 = 2
                            voiceType2 = 1
                            start2 = 0.7 #  set delay between voice 1 and 2
                            speed2 = 0.7
                            segstart2, segend2 = wav_segments[(soundfile2, word_id2)]
                            #segstart2 = 0 # segment start and end within sound file
                            #segend2 = 0 # if zero, play whole file
                            amp2 = -3 # voice amplitude in dB
                            dur2, maxamp2 = utils.getSoundParmFromFile(soundfile2)
                            sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel2, voiceType2, start2, soundfile2, speed2, segstart2, segend2, amp2, maxamp2))
                            nextTime2 += (dur2/speed2)
                            enableVoice2 = 0
                        # trig another word in voice 2 only if word 2 has finished playing (and sync to start of voice 1)
                        if nextTime1 > nextTime2: enableVoice2 = 1 

                        face_id = np.random.choice(sound_to_face[word_id])
                        video_time = (1000*dur)/IO.VIDEO_SAMPLE_TIME
                        stride = int(np.ceil(NAP.shape[0]/video_time))
                        projection = video_producer[(audio_id, face_id)](NAP[::stride])

                        for row in projection:
                            utils.send_array(projector, np.resize(row, FRAME_SIZE[::-1]))

                        # as the first crude method of assembling a sentence, just wait for the word duration here
                        time.sleep(dur)

                except:
                    utils.print_exception('Sentence response aborted.')


def learn(host):
    import Oger

    mic, speaker, camera, projector, face, stateQ, eventQ, sender, state, poller, me, brainQ = _connect(host)
    
    audio = deque()
    video = deque()
    faces = deque()

    audio_first_segment = []
    video_first_segment = []
    wav_first_segment = []
    
    NAPs = []
    wavs = []
    wav_segments = {}
    face_history = []
    face_hashes = []
    NAP_hashes = []

    sound_to_face = {}
    face_to_sound = {}
    
    audio_recognizer = []
    video_recognizer = []
    face_recognizer = []
    audio_producer = []
    video_producer = {}

    maxlen = []
    maxlen_scaled = []
        
    # WRITE IMAGES TO DISK!
    while True:
        events = dict(poller.poll())
        
        if stateQ in events:
            state = stateQ.recv_json()

        if mic in events:
            new_audio = utils.recv_array(mic)
            if state['record']:
                audio.append(new_audio)
            
        if camera in events:
            new_video = utils.recv_array(camera)
            if state['record']:
                frame = cv2.resize(new_video, FRAME_SIZE)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.
                gray_flattened = np.ndarray.flatten(gray_image)
                video.append(gray_flattened)
        
        if face in events:
            new_face = utils.recv_array(face)
            gray = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY) / 255.

            if state['record']:
                faces.append(gray)

            if state['facerecognition']:
                try: 
                    face_id = face_recognizer.predict(np.ndarray.flatten(gray))[0]
                    print 'Face {} has previously said {}'.format(face_id, face_to_sound[face_id])
                                        
                except Exception, e:
                    print e, 'Face recognition aborted.'

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'learn' in pushbutton and pushbutton['learn'] == me.name:
                try:
                    backup = [ audio_recognizer, face_recognizer, maxlen, maxlen_scaled ]
                    
                    filename = pushbutton['filename']
                    if len(wav_first_segment):
                        print 'Learning to associate {} -> {}'.format(wav_first_segment, filename)
                        filename = wav_first_segment

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
                            predicted_faces = [ face_recognizer.predict(np.ndarray.flatten(f))[0] for f in new_faces ]
                            uniq = np.unique(predicted_faces)
                            face_id = uniq[np.argsort([ sum(predicted_faces == u) for u in uniq ])[-1]]
                            hammings = [ utils.hamming_distance(f, m) for f in new_faces_hashes for m in face_hashes[face_id] ]

                        if np.mean(hammings) < HAMMERTIME:
                            face_history[face_id].extend(new_faces)
                            face_hashes[face_id].extend(new_faces_hashes)
                            print 'Face is similar to face {}, hamming mean {}, has previously said {}'.format(face_id, np.mean(hammings), face_to_sound[face_id])
                        else:
                            print 'New face, hamming mean {} from face {}'.format(np.mean(hammings), face_id)
                            face_history.append(new_faces)
                            face_hashes.append(new_faces_hashes)
                            face_id = len(face_history) - 1
                            face_to_sound[face_id] = []

                        if len(face_history) > 1:
                            x_train = [ np.ndarray.flatten(f) for cluster in face_history for f in cluster ]
                            face_recognizer = svm.LinearSVC()
                            targets = [ i for i,f in enumerate(face_history) for _ in f ]
                            face_recognizer.fit(x_train, targets)
                    else:
                        print 'Face not detected.'
                        
                    audio_segments = utils.get_segments(utils.wait_for_wav(filename))
                    
                    print 'Learning {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    start_time = time.time()
                    new_sentence = cochlear(filename)
                    print 'Calculating cochlear neural activation patterns took {} seconds'.format(time.time() - start_time)

                    filename = pushbutton['filename'] if len(wav_first_segment) else filename
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')
                    
                    for segment, new_sound in enumerate([ utils.trim_right(utils.scale(new_sentence[norm_segments[i]:norm_segments[i+1]])) for i in range(len(norm_segments)-1) ]):
                        # Do we know this sound?
                        hammings = [ np.inf ]
                        new_audio_hash = utils.d_hash(new_sound)
                        audio_id = 0
                        if len(NAPs) == 1:
                            hammings = [ utils.hamming_distance(new_audio_hash, h) for h in NAP_hashes[0] ]

                        if audio_recognizer:
                            resampled_new_sound = utils.zero_pad(resample(new_sound, float(maxlen)/new_sound.shape[0], 'sinc_best'), maxlen_scaled)
                            resampled_flattened_new_sound = np.ndarray.flatten(resampled_new_sound)
                            audio_id = audio_recognizer.predict(resampled_flattened_new_sound)[0]
                            hammings = [ utils.hamming_distance(new_audio_hash, h) for h in NAP_hashes[audio_id] ]

                        if np.mean(hammings) < HAMMERTIME:
                            NAPs[audio_id].append(new_sound)
                            NAP_hashes[audio_id].append(new_audio_hash)
                            wavs[audio_id].append(filename)
                            print 'Sound is similar to sound {}, hamming mean {}, previously said by faces {}'.format(audio_id, np.mean(hammings), sound_to_face[audio_id])
                        else:
                            print 'New sound, hamming mean {} from sound {}'.format(np.mean(hammings), audio_id)
                            NAPs.append([new_sound])
                            NAP_hashes.append([new_audio_hash])
                            wavs.append([filename])
                            audio_id = len(NAPs) - 1
                            sound_to_face[audio_id] = []

                        # The mapping from wavfile and audio ID to the segment within the audio file
                        wav_segments[(filename, audio_id)] = [ audio_segments[segment], audio_segments[segment+1] ]
                        
                        # Scale the sizes of the samples according to the biggest one. The idea is that this scale well. Otherwise, create overlapping bins.
                        start_time = time.time()
                        maxlen = max([ m.shape[0] for memory in NAPs for m in memory ])
                        resampled_memories = [ [ resample(m, float(maxlen)/m.shape[0], 'sinc_best') for m in memory ] for memory in NAPs ]
                        maxlen_scaled = max([ m.shape[0] for memory in resampled_memories for m in memory ])

                        if len(NAPs) > 1:
                            resampled_memories = [ [ utils.zero_pad(m, maxlen_scaled) for m in memory ] for memory in resampled_memories ]
                            resampled_flattened_memories = [ np.ndarray.flatten(m) for memory in resampled_memories for m in memory ]
                            audio_targets = [ i for i,f in enumerate(NAPs) for _ in f ]

                            audio_recognizer = svm.LinearSVC()
                            # Is the high dimensionality required? Maybe PCA could help reduce the training time necessary. 
                            audio_recognizer.fit(resampled_flattened_memories, audio_targets)

                        if not face_id in sound_to_face[audio_id]:
                            sound_to_face[audio_id].append(face_id)
                        if face_id is not -1 and not audio_id in face_to_sound[face_id]:
                            face_to_sound[face_id].append(audio_id)
                        
                    # Send sound id and classification data to associations analysis
                    similar_ids = [ utils.hamming_distance(new_audio_hash, np.random.choice(h)) for h in NAP_hashes ]
                    association.analyze(filename,audio_id,wavs,similar_ids,sound_to_face,face_to_sound)

                    video_segment = np.array(list(video))
                    if video_segment.shape[0] == 0:
                        video_segment = np.array([ np.ndarray.flatten(np.zeros(FRAME_SIZE)) for _ in range(10) ])
                        print 'No video recorded. Using black image as stand-in.'

                    NAP_len = new_sentence.shape[0]
                    video_len = video_segment.shape[0]
                    stride = int(max(1,np.floor(float(NAP_len)/video_len)))
                    
                    x = new_sound[:NAP_len - np.mod(NAP_len, stride*video_len):stride]
                    if x.shape[0] == 0:
                        x = new_sound[::stride]
                    y = video_segment[:x.shape[0]]

                    tarantino = train_network(x,y)
                    tarantino.stride = stride
                    video_producer[(audio_id, face_id)] = tarantino

                    brainQ.send_pyobj([ wavs, wav_segments, sound_to_face, face_to_sound, audio_recognizer, video_producer, maxlen, maxlen_scaled ])
                    print 'Learning classifier and video network in {} seconds'.format(time.time() - start_time)                    
                except:
                    utils.print_exception('Learning aborted. Backing up.')
                    audio_recognizer, face_recognizer, maxlen, maxlen_scaled = backup

                pushbutton['reset'] = True

            if 'play_id' in pushbutton:
                try:
                    items = pushbutton['play_id'].split(' ')
                    play_audio_id = int(items[0])
                    voiceChannel = int(items[1])
                    print 'play_audio_id', play_audio_id, 'voice', voiceChannel
                    print 'wavs[play_audio_id]', wavs[play_audio_id]
                    print wavs
                    soundfile = np.random.choice(wavs[play_audio_id])
                    voiceChannel = 1
                    voiceType = 1
                    speed = 1
                    segstart, segend = wav_segments[(soundfile, audio_id)]
                    #segstart = 0 # segment start and end within sound file
                    #segend = 0 # if zero, play whole file
                    amp = -3 # voice amplitude in dB
                    dur, maxamp = utils.getSoundParmFromFile(soundfile)
                    start = 0
                    sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType, start, response, speed, segstart, segend, amp, maxamp))
                except:
                    utils.print_exception('play_id aborted.')


            if 'reset' in pushbutton:
                audio.clear()
                video.clear()
                faces.clear()
                audio_first_segment = []
                video_first_segment = []
                wav_first_segment = []
            
            if 'setmarker' in pushbutton:
                wav_first_segment = pushbutton['setmarker']
                audio_first_segment = np.array(list(audio))
                video_first_segment = np.array(list(video))
                print 'FACES NEW FIRST SEGMENT FIXME'
                audio.clear()
                video.clear()
                faces.clear()

            if 'showme' in pushbutton:
                # just for inspecting the contents of objects while running 
                print 'printing '+pushbutton['showme']
                o = compile('print '+pushbutton['showme'], '<string>','exec')
                eval(o)

            if 'load' in pushbutton:
                filename = pushbutton['load']
                audio_recognizer, audio_producer, video_producer, NAPs, wavs, maxlen,\
                            sound_to_face,\
                            face_to_sound,\
                            association.wavs_as_words,\
                            association.wordTime,\
                            association.time_word,\
                            association.duration_word,\
                            association.similarWords,\
                            association.neighbors,\
                            association.neighborAfter,\
                            association.wordFace,\
                            association.faceWord\
                            = pickle.load(file(filename, 'r'))
                print 'Brain loaded from file {} ({})'.format(filename, utils.filesize(filename))

            # CHECK WHAT TAKES UP SO MUCH SPACE! I suspect the video_producer matrix.
            if 'save' in pushbutton:
                filename = '{}.{}'.format(pushbutton['save'], me.name)
                pickle.dump((audio_recognizer, audio_producer, video_producer, NAPs, wavs, maxlen,
                sound_to_face,
                face_to_sound,
                association.wavs_as_words,
                association.wordTime,
                association.time_word,
                association.duration_word,
                association.similarWords,
                association.neighbors,
                association.neighborAfter,
                association.wordFace,
                association.faceWord)
                , file(filename, 'w'))
                print '{} saved as file {} ({})'.format(me.name, filename, utils.filesize(filename))
                

def _connect(host):
    me = mp.current_process()
    me.name = 'BRAIN{}'.format(str(uuid1()))
    print '{} PID {} connecting to {}'.format(me.name, me.pid, host)

    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, IO.MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    speaker = context.socket(zmq.PUSH)
    speaker.connect('tcp://{}:{}'.format(host, IO.SPEAKER)) 

    camera = context.socket(zmq.SUB)
    camera.connect('tcp://{}:{}'.format(host, IO.CAMERA))
    camera.setsockopt(zmq.SUBSCRIBE, b'')

    projector = context.socket(zmq.PUSH)
    projector.connect('tcp://{}:{}'.format(host, IO.PROJECTOR)) 

    face = context.socket(zmq.SUB)
    face.connect('tcp://{}:{}'.format(host, IO.FACE))
    face.setsockopt(zmq.SUBSCRIBE, b'')

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, IO.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))
    sender.send_json('register {} {}'.format(me.name, mp.cpu_count()))

    snapshot = context.socket(zmq.REQ)
    snapshot.connect('tcp://{}:{}'.format(host, IO.SNAPSHOT))
    snapshot.send(b'Send me the state, please')
    state = snapshot.recv_json()

    brainQ = context.socket(zmq.PUB)
    brainQ.bind('tcp://*:{}'.format(IO.BRAIN))
        
    poller = zmq.Poller()
    poller.register(mic, zmq.POLLIN)
    poller.register(camera, zmq.POLLIN)
    poller.register(face, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    return mic, speaker, camera, projector, face, stateQ, eventQ, sender, state, poller, me, brainQ
                        
if __name__ == '__main__':
    classifier_brain(sys.argv[1] if len(sys.argv) == 2 else 'localhost')
