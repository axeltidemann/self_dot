import multiprocessing as mp
from uuid import uuid4
from collections import deque
import sys
import glob
import cPickle as pickle
from subprocess import call
import time
import os
import itertools
import random
from collections import namedtuple

import numpy as np
import zmq
from scipy.stats import itemfreq
from scipy.cluster.vq import kmeans, vq

import utils
import IO
import association
import my_sai_test as mysai
import myCsoundAudioOptions

from brain import _project, _extract_NAP, _three_amigos

AUDIO_HAMMERTIME = 9
FRAME_SIZE = (160,120) # Neural network image size, 1/4 of full frame size.

AudioSegment = namedtuple('AudioSegment', ['audio_id', 'crude_hash', 'fine_hash', 'wav_file', 'segment_idxs']) #should be times

class AudioMemory:
    def __init__(self):
        self.num_bins = 11
        self.clean = np.linspace(0, 200, self.num_bins) # Effectively 10 second limit, .5 second overlaps.
        self.overlap = np.linspace(10, 210, self.num_bins)
        self.NAP_intervals = {}
        self.audio_ids = {}
        self.audio_id_counter = 0

    def _digitize(self, NAP):
        idx_clean = np.digitize([ NAP.shape[0] ], self.clean)[0]
        idx_overlap = np.digitize([ NAP.shape[0] ], self.overlap)[0]

        clean_key = (self.clean[idx_clean - 1], self.clean[min(self.num_bins-1,idx_clean)])
        overlap_key = (self.overlap[idx_overlap - 1], self.overlap[min(self.num_bins-1,idx_overlap)])

        return clean_key, overlap_key

    def _insert(self, D, key, value):
        if key in D:
            D[key].append(value)
        else:
            D[key] = [ value ]
    
    def find(self, NAP):
        crude_hash = utils.d_hash(NAP, hash_size=8)
        fine_hash = utils.d_hash(NAP, hash_size=16)
        clean_key, overlap_key = self._digitize(NAP)

        unsorted = self.NAP_intervals[clean_key] if clean_key in self.NAP_intervals else [] + self.NAP_intervals[overlap_key] if overlap_key in self.NAP_intervals else []

        return sorted(unsorted, key = lambda x: utils.hamming_distance(fine_hash, x.fine_hash))[0] if len(unsorted) else [], crude_hash, fine_hash, clean_key, overlap_key
    
    def learn(self, NAP, wav_file, segment_idxs):
        best_match, crude_hash, fine_hash, clean_key, overlap_key = self.find(NAP)
        
        if not len(best_match):
            audio_id = self.audio_id_counter
            print 'New audio_id {}, never before heard length {}'.format(audio_id, NAP.shape[0])
            self.audio_id_counter += 1
        else:
            if utils.hamming_distance(crude_hash, best_match.crude_hash) < AUDIO_HAMMERTIME:
                audio_id = best_match.audio_id
                print 'Similar to audio_id {}, hamming distance {}'.format(audio_id, utils.hamming_distance(crude_hash, best_match.crude_hash))
            else:
                audio_id = self.audio_id_counter
                self.audio_id_counter += 1                
                print 'New audio_id {}, hamming distance {} from audio_id {}'.format(audio_id, utils.hamming_distance(crude_hash, best_match.crude_hash), best_match.audio_id)
                
        audio_segment = AudioSegment(audio_id, crude_hash, fine_hash, wav_file, segment_idxs)

        self._insert(self.NAP_intervals, clean_key, audio_segment)
        self._insert(self.NAP_intervals, overlap_key, audio_segment)
        self._insert(self.audio_ids, audio_id, audio_segment)
        
        return audio_id

    def all_segments(self):
        return [ audio_segment for key, value in self.audio_ids.iteritems() for audio_segment in value ]

    def forget(self, audio_segment):
        self.audio_ids[audio_segment.audio_id].remove(audio_segment)
        for _, audio_segments in self.NAP_intervals.iteritems():
            try:
                audio_segments.remove(audio_segment)
            except:
                continue
        self._cleanse_keys()
        
    def _cleanse_keys(self):
        for empty_key in [ key for key, value in self.audio_ids.iteritems() if len(value) == 0 ]:
            del self.audio_ids[empty_key]
        for empty_key in [ key for key, value in self.NAP_intervals.iteritems() if len(value) == 0 ]:
            del self.NAP_intervals[empty_key]
        
def new_respond(control_host, learn_host, debug=False):
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

    counterQ = context.socket(zmq.REQ)
    counterQ.connect('tcp://{}:{}'.format(control_host, IO.COUNTER))
    
    cognitionQ = context.socket(zmq.PUSH)
    cognitionQ.connect('tcp://{}:{}'.format(control_host, IO.COGNITION))

    association = context.socket(zmq.REQ)
    association.connect('tcp://{}:{}'.format(learn_host, IO.ASSOCIATION))

    snapshot = context.socket(zmq.REQ)
    snapshot.connect('tcp://{}:{}'.format(control_host, IO.SNAPSHOT))

    scheduler = context.socket(zmq.PUSH)
    scheduler.connect('tcp://{}:{}'.format(control_host, IO.SCHEDULER))

    dreamQ = context.socket(zmq.PULL)
    dreamQ.bind('tcp://*:{}'.format(IO.DREAM))

    snapshot.send_json('Give me state!')
    state = snapshot.recv_json()

    poller = zmq.Poller()
    poller.register(eventQ, zmq.POLLIN)
    poller.register(brainQ, zmq.POLLIN)
    poller.register(dreamQ, zmq.POLLIN)

    sound_to_face = []
    wordFace = {}
    face_to_sound = []
    faceWord = {}
    register = {}
    video_producer = {}
    voiceType1 = 1
    voiceType2 = 6
    wordSpace1 = 0.3
    wordSpaceDev1 = 0.3
    wordSpace2 = 0.1
    wordSpaceDev2 = 0.3

    audio_ids = []
    wavs = []
    wav_audio_ids = []
    NAP_hashes = {}
    most_significant_audio_id = []
    
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
                _, _, audio_ids, audio_memory, most_significant_audio_id, wavs, wav_audio_ids = register[wav_file][0]
                _, _, tarantino = register[wav_file][1]
                _, _, face_id, face_recognizer = register[wav_file][2]          
                print 'Audio - video - face recognizers related to {} arrived at responder, total processing time {} seconds'.format(wav_file, time.time() - utils.filetime(wav_file))

                for audio_id in audio_ids: # If audio_ids is empty, none of this will happen
                    video_producer[(audio_id, face_id)] = tarantino 
                    if audio_id < len(sound_to_face) and not face_id in sound_to_face[audio_id]: # sound heard before, but not said by this face 
                        sound_to_face[audio_id].append(face_id)
                    if audio_id == len(sound_to_face):
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
                    print 'face_id for audio segment learned', face_id
                    if face_id is not -1:
                        if face_id < len(face_to_sound) and not audio_id in face_to_sound[face_id]: #face seen before, but the sound is new
                            face_to_sound[face_id].append(audio_id)
                        if face_id == len(face_to_sound):
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
                for audio_id in audio_ids:
                    # I SUSPECT THIS IS WRONG, SINCE THERE IS NO SORTING OF THESE HAMMING DISTANCES IN ASSOCIATION.PY
                    new_audio_hash = audio_memory.audio_ids[audio_id][-1].crude_hash
                    similar_ids_for_this_audio_id = [ utils.hamming_distance(new_audio_hash, random.choice(h).crude_hash) for h in audio_memory.audio_ids.itervalues() ]
                    similar_ids.append(similar_ids_for_this_audio_id)

                if len(audio_ids):
                    association.send_pyobj(['analyze',wav_file,wav_audio_ids,audio_ids,wavs,similar_ids,wordFace,faceWord])
                    association.recv_pyobj()
                    sender.send_json('last_most_significant_audio_id {}'.format(most_significant_audio_id))

                cognitionQ.send_pyobj(face_recognizer) # A possiblity of recognizing a face that is not connecting to any soundfiles

                                
        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'respond_single' in pushbutton:
                try:
                    filename = pushbutton['filename']
                    audio_segments = utils.get_segments(filename)
                    print 'Single response to {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    new_sentence = utils.csv_to_array(filename + 'cochlear')
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    segment_id = utils.get_most_significant_word(filename)

                    NAP = utils.trim_right(new_sentence[norm_segments[segment_id]:norm_segments[segment_id+1]])
           
                    if debug:            
                        plt.imshow(NAP.T, aspect='auto')
                        plt.draw()

                    best_match,_,_,_,_ = audio_memory.find(NAP)
                    soundfile = best_match.wav_file
                    segstart, segend = best_match.segment_idxs

                    voiceChannel = 1
                    speed = 1
                    amp = -3 # voice amplitude in dB
                    _,dur,maxamp,_ = utils.getSoundInfo(soundfile)
                    
                    start = 0
                    voice1 = 'playfile {} {} {} {} {} {} {} {} {}'.format(1, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp)
                    voice2 = ''

                    print 'Recognized as sound {}'.format(best_match.audio_id)

                    # sound_to_face, video_producer
                    projection = _project(best_match.audio_id, sound_to_face, NAP, video_producer)

                    scheduler.send_pyobj([[ dur, voice1, voice2, projection, FRAME_SIZE ]])
                    print 'Respond time from creation of wav file was {} seconds'.format(time.time() - utils.filetime(filename))
                except:
                    utils.print_exception('Single response aborted.')


            if 'play_sentence' in pushbutton:
                try:
                    sentence = pushbutton['sentence']
                    sentence = eval(sentence)
                    print '*** (play) Play sentence', sentence
                    start = 0 
                    nextTime1 = 0
                    play_events = []
                    for i in range(len(sentence)):
                        word_id = sentence[i]
                        soundfile = np.random.choice(wavs[word_id])
                        speed = 1

                        segstart, segend = wav_audio_ids[(soundfile, word_id)]
                        NAP = _extract_NAP(segstart, segend, soundfile)

                        amp = -3 # voice amplitude in dB
                        _,totaldur,maxamp,_ = utils.getSoundInfo(soundfile)
                        dur = segend-segstart
                        if dur <= 0: dur = totaldur
                        # play in both voices
                        voice1 = 'playfile {} {} {} {} {} {} {} {} {}'.format(1, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp)
                        voice2 = 'playfile {} {} {} {} {} {} {} {} {}'.format(2, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp)
                        wordSpacing1 = wordSpace1 + np.random.random()*wordSpaceDev1
                        print 'PLAY RESPOND SPACING', wordSpacing1
                        nextTime1 += (dur/speed)+wordSpacing1

                        projection = _project(audio_id, sound_to_face, NAP, video_producer)

                        play_events.append([ dur+wordSpacing1, voice1, voice2, projection, FRAME_SIZE ])                        
                    scheduler.send_pyobj(play_events)
                except:
                    utils.print_exception('Sentence play aborted.')

            if 'respond_sentence' in pushbutton:
                print 'SENTENCE Respond to', pushbutton['filename'][-12:]
                    
                try:
                    filename = pushbutton['filename']
                    audio_segments = utils.get_segments(filename)
                    print 'Sentence response to {} duration {} seconds with {} segments'.format(filename, audio_segments[-1], len(audio_segments)-1)
                    new_sentence = utils.csv_to_array(filename + 'cochlear')
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    segment_id = utils.get_most_significant_word(filename)
                    print '**Sentence selected to respond to segment {}'.format(segment_id)

                    NAP = utils.trim_right(new_sentence[norm_segments[segment_id]:norm_segments[segment_id+1]])

                    best_match,_,_,_,_ = audio_memory.find(NAP)
                    audio_id = best_match.audio_id
                    soundfile = best_match.wav_file
        
                    numWords = len(audio_segments)-1
                    print numWords
                    association.send_pyobj(['setParam', 'numWords', numWords ])
                    association.recv_pyobj()
                    
                    association.send_pyobj(['makeSentence', audio_id])
                    print 'respond_sentence waiting for association output...', 
                    sentence, secondaryStream = association.recv_pyobj()

                    print '*** (respond) Play sentence', sentence, secondaryStream
                    start = 0 
                    nextTime1 = 0
                    nextTime2 = 0
                    enableVoice2 = 1

                    play_events = []

                    for i in range(len(sentence)):
                        word_id = sentence[i]
                        soundfile = np.random.choice(wavs[word_id])
                        voiceChannel = 1
                        speed = 1
                        
                        # segment start and end within sound file, if zero, play whole file
                        segstart, segend = wav_audio_ids[(soundfile, word_id)]
                        NAP = _extract_NAP(segstart, segend, soundfile)
                        
                        amp = -3 # voice amplitude in dB
                        #totaldur, maxamp = utils.getSoundParmFromFile(soundfile)
                        _,totaldur,maxamp,_ = utils.getSoundInfo(soundfile)
                        dur = segend-segstart
                        if dur <= 0: dur = totaldur
                        voice1 = 'playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType1, start, soundfile, speed, segstart, segend, amp, maxamp)
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
                            try:
                                segstart2, segend2 = wav_audio_ids[(soundfile2, word_id2)]
                                dur2 = segend2-segstart2
                                #totalDur2, maxamp2 = utils.getSoundParmFromFile(soundfile2)
                                _,totalDur2,maxamp2,_ = utils.getSoundInfo(soundfile)
                                if dur2 <= 0: dur2 = totalDur2
                                voice2 = 'playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel2, voiceType2, start2, soundfile2, speed2, segstart2, segend2, amp2, maxamp2)
                                wordSpacing2 = wordSpace2 + np.random.random()*wordSpaceDev2
                                nextTime2 += (dur2/speed2)+wordSpacing2
                            except:
                                voice2 = ''
                                utils.print_exception('VOICE 2 tried to access an illegal soundfile/audio_id combination.')
                            #enableVoice2 = 0
                        # trig another word in voice 2 only if word 2 has finished playing (and sync to start of voice 1)
                        if nextTime1 > nextTime2: enableVoice2 = 1 

                        projection = _project(audio_id, sound_to_face, NAP, video_producer)
                        print 'SENTENCE RESPOND SPACING', wordSpacing1
                        play_events.append([ dur+wordSpacing1, voice1, voice2, projection, FRAME_SIZE ])

                    scheduler.send_pyobj(play_events)
                    print 'Sentence respond time from creation of wav file was {} seconds'.format(time.time() - utils.filetime(filename))
                except:
                    utils.print_exception('Sentence response aborted.')
                    
            if 'testSentence' in pushbutton:
                print 'testSentence', pushbutton
                association.send_pyobj(['makeSentence',int(pushbutton['testSentence'])])
                print 'testSentence waiting for association output...'
                sentence, secondaryStream = association.recv_pyobj()
                print '*** Test sentence', sentence, secondaryStream
            
            if 'assoc_setParam' in pushbutton:
                try:
                    parm, value = pushbutton['assoc_setParam'].split()
                    association.send_pyobj(['setParam', parm, value ])
                    association.recv_pyobj()
                except:
                    utils.print_exception('Assoc set param aborted.')

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
                    #print 'wav_audio_ids', wav_audio_ids
                    segstart, segend = wav_audio_ids[(soundfile, play_audio_id)]
                    #segstart = 0 # segment start and end within sound file
                    #segend = 0 # if zero, play whole file
                    amp = -3 # voice amplitude in dB
                    #dur, maxamp = utils.getSoundParmFromFile(soundfile)
                    _,dur,maxamp,_ = utils.getSoundInfo(soundfile)
                    start = 0
                    sender.send_json('playfile {} {} {} {} {} {} {} {} {}'.format(voiceChannel, voiceType, start, soundfile, speed, segstart, segend, amp, maxamp))
                except:
                    utils.print_exception('play_id aborted.')

            if 'print_me' in pushbutton:
                # just for inspecting the contents of objects while running 
                print 'printing '+pushbutton['print_me']
                if 'brain ' in pushbutton['print_me']: 
                    print_variable = pushbutton['print_me'].split('brain ')[-1]
                    try:
                        print eval(print_variable)
                    except Exception, e:
                        print e, 'print_me in brain failed.'
                elif 'association ' in pushbutton['print_me']: 
                    print_variable = pushbutton['print_me'].split('association ')[-1]
                    association.send_pyobj(['print_me',print_variable])

            if 'dream' in pushbutton:
                play_events = []
                for audio_segment in audio_memory.all_segments():
                    segstart, segend = audio_segment.segment_idxs
                    dur = segend - segstart
                    NAP = _extract_NAP(segstart, segend, audio_segment.wav_file)
                    speed = 1
                    amp = -3
                    maxamp = 1
                    start = 0
                    voice1 = 'playfile {} {} {} {} {} {} {} {} {}'.format(1, 6, np.random.rand()/3, audio_segment.wav_file, speed, segstart, segend, amp, maxamp)
                    projection = _project(audio_segment.audio_id, sound_to_face, NAP, video_producer)
                    voice2 = 'playfile {} {} {} {} {} {} {} {} {}'.format(2, 6, np.random.randint(3,6), audio_segment.wav_file, speed, segstart, segend, amp, maxamp)
                    play_events.append([ dur, voice1, voice2, projection, FRAME_SIZE ])
                print 'Dream mode playing back {} memories'.format(len(play_events))
                scheduler.send_pyobj(play_events)

            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], mp.current_process().name), [ sound_to_face, wordFace, face_to_sound, faceWord, video_producer, wavs, wav_audio_ids, audio_classifier, maxlen, NAP_hashes, face_id, face_recognizer, audio_memory ])

            if 'load' in pushbutton:
                sound_to_face, wordFace, face_to_sound, faceWord, video_producer, wavs, wav_audio_ids, audio_classifier, maxlen, NAP_hashes, face_id, face_recognizer, audio_memory = utils.load('{}.{}'.format(pushbutton['load'], mp.current_process().name))


def new_learn_audio(host, debug=False):
    context = zmq.Context()

    mic = context.socket(zmq.SUB)
    mic.connect('tcp://{}:{}'.format(host, IO.MIC))
    mic.setsockopt(zmq.SUBSCRIBE, b'')

    dreamQ = context.socket(zmq.PUSH)
    dreamQ.connect('tcp://{}:{}'.format(host, IO.DREAM))


    stateQ, eventQ, brainQ = _three_amigos(context, host)

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, IO.EXTERNAL))

    counterQ = context.socket(zmq.REQ)
    counterQ.connect('tcp://{}:{}'.format(host, IO.COUNTER))
    
    poller = zmq.Poller()
    poller.register(mic, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    audio = deque()
    NAPs = []
    wavs = []
    wav_audio_ids = {}
    NAP_hashes = {}

    audio_classifier = []
    audio_recognizer = []
    global_audio_recognizer = []
    mixture_audio_recognizer = []
    maxlen = []

    deleted_ids = []
    
    state = stateQ.recv_json()
    
    black_list = open('black_list.txt', 'a')

    audio_memory = AudioMemory()
    
    if debug:
        import matplotlib.pyplot as plt
        plt.ion()

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
                    new_sentence = utils.csv_to_array(filename + 'cochlear')
                    norm_segments = np.rint(new_sentence.shape[0]*audio_segments/audio_segments[-1]).astype('int')

                    audio_ids = []
                    new_audio_hash = []
                    amps = utils.get_amps(filename)
                    most_significant_value = -np.inf
                    most_significant_audio_id = []

                    original_NAP_length = len(NAPs)
                    
                    for segment, new_sound in enumerate([ utils.trim_right(new_sentence[norm_segments[i]:norm_segments[i+1]]) for i in range(len(norm_segments)-1) ]):
                        # We filter out short, abrupt sounds with lots of noise.
                        if np.mean(new_sound) < 2 or new_sound.shape[0] == 0:
                          black_list.write('{} {}\n'.format(filename, segment))
                          print 'BLACKLISTED segment {} in file {}'.format(segment, filename)
                          continue

                        if debug:
                            utils.plot_NAP_and_energy(new_sound, plt)

                        audio_id = audio_memory.learn(new_sound, filename, [ audio_segments[segment], audio_segments[segment+1] ])

                        # START LEGACY
                        try:
                            wavs[audio_id].append(filename)
                        except:
                            wavs.append([filename])
                        wav_audio_ids[(filename, audio_id)] = [ audio_segments[segment], audio_segments[segment+1] ]
                        # END LEGACY
                        
                        audio_ids.append(audio_id)
                        if amps[segment] > most_significant_value:
                            most_significant_audio_id = audio_id
                            most_significant_value = amps[segment]

                    black_list.flush()
                    print 'AUDIO IDs after blacklisting {}'. format(audio_ids)
                    if len(audio_ids):
                        # while len(NAPs) - len(deleted_ids) > AUDIO_MEMORY_SIZE:
                        #     utils.delete_loner(counterQ, NAPs, 'audio_ids_counter', int(AUDIO_MEMORY_SIZE*PROTECT_PERCENTAGE), deleted_ids)

                        # maxlen = max([ m.shape[0] for memory in NAPs for m in memory if len(m) ])
                        # memories = [ np.ndarray.flatten(utils.zero_pad(m, maxlen)) for memory in NAPs for m in memory if len(m) ]

                        # targets = [ i for i,f in enumerate(NAPs) for k in f if len(k) ]
                        # audio_classifier = train_rPCA_SVM(memories, targets)

                        # all_hammings = [ utils.hamming_distance(new_audio_hash[i], new_audio_hash[j])
                        #                                         for i in range(len(new_audio_hash)) for j in range(len(new_audio_hash)) if i > j ]
                    
                        # print 'RHYME VALUE', np.mean(sorted(all_hammings)[int(len(all_hammings)/2):])
                        # rhyme = np.mean(sorted(all_hammings)[int(len(all_hammings)/2):]) < RHYME_HAMMERTIME

                        # sender.send_json('rhyme {}'.format(rhyme))

                        brainQ.send_pyobj(['audio_learn', filename, audio_ids, audio_memory, most_significant_audio_id, wavs, wav_audio_ids])
                        print 'Audio learned from {} in {} seconds'.format(filename, time.time() - t0)
                    else:
                        print 'SKIPPING fully blacklisted file {}'.format(filename)
                except:
                    utils.print_exception('Audio learning aborted.')

                audio.clear()

            if 'dream' in pushbutton:
                new_dream(audio_memory)
                     
            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], mp.current_process().name), [ deleted_ids, NAPs, wavs, wav_audio_ids, NAP_hashes, audio_classifier, maxlen, audio_memory ])
                
            if 'load' in pushbutton:
                deleted_ids, NAPs, wavs, wav_audio_ids, NAP_hashes, audio_classifier, maxlen, audio_memory = utils.load('{}.{}'.format(pushbutton['load'], mp.current_process().name))


def new_dream(audio_memory):
    
    #import matplotlib.pyplot as plt
    #plt.ion()

    try:
        print 'Dreaming - removing wrongly binned filenames'
        mega_filenames_and_indexes = []

        for audio_id, audio_segments in audio_memory.audio_ids.iteritems():

            NAP_detail = 'low'
            filenames_and_indexes = []

            for audio_segment in audio_segments:
                segstart, segend = audio_segment.segment_idxs
                audio_times = utils.get_segments(audio_segments.wav_file)
                norm_segstart = segstart/audio_times[-1]
                norm_segend = segend/audio_times[-1]
                filenames_and_indexes.append([ soundfile, norm_segstart, norm_segend, audio_id, NAP_detail ])
                
            mega_filenames_and_indexes.extend(filenames_and_indexes)

            k = 2
            print 'Examining audio_id {}'.format(audio_id)
            if len(audio_segments) == 1:
                print 'Just one member in this audio_id, skipping analysis'
                continue

            sparse_codes = mysai.experiment(filenames_and_indexes, k)
            # plt.matshow(sparse_codes, aspect='auto')
            # plt.colorbar()
            # plt.draw()

            coarse = np.mean(sparse_codes, axis=1)
            coarse.shape = (len(coarse), 1)

            codebook,_ = kmeans(coarse, k)
            instances = [ vq(np.atleast_2d(s), codebook)[0] for s in coarse ]

            freqs = itemfreq(instances)
            sorted_freqs = sorted(freqs, key=lambda x: x[1])
            print 'Average sparse codes: {} Class count: {}'.format(list(itertools.chain.from_iterable(coarse)), sorted_freqs)

            if len(sorted_freqs) == 1:
                print 'Considered to be all the same.'
                continue

            fewest_class = sorted_freqs[0][0]
            ousted_audio_segments = [ audio_segment for audio_segment, i in zip(audio_segments, instances) if i == fewest_class ]
            print 'Class {} has fewest members, deleting audio_segments {}'.format(fewest_class, ousted_audio_segments)
            filter(audio_memory.forget, ousted_audio_segments)

        print 'Creating mega super self-organized class'

        for row in mega_filenames_and_indexes:
            row[-1] = 'high'

        high_resolution_k = 256
        clusters = 24
        sparse_codes = mysai.experiment(mega_filenames_and_indexes, high_resolution_k)
        sparse_codes = np.array(sparse_codes)
        # plt.matshow(sparse_codes, aspect='auto')
        # plt.colorbar()
        # plt.draw()

        codebook,_ = kmeans(sparse_codes, clusters)
        instances = [ vq(np.atleast_2d(s), codebook)[0] for s in sparse_codes ]

        cluster_list = {}
        for mega, instance in zip(mega_filenames_and_indexes, instances):
            soundfile,_,_,audio_id,_ = mega
            cluster_list[(soundfile, audio_id)] = instance

        print cluster_list
    except:
        utils.print_exception('NIGHTMARE!')
