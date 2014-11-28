#!/usr/bin/python
# -*- coding: latin-1 -*-

import pickle
import glob
import os
NUMBER_OF_BRAINS = 5

def filesize(filename):
    return bytes2human(os.path.getsize(filename))
    
def bytes2human(n, format="%(value)i%(symbol)s"):
    """
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    """
    symbols = ('b', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)

def find_last_valid_brain():
    files = glob.glob('BRAIN_*')
    files.sort(key=os.path.getmtime, reverse=True)
    for f in files:
        stem = f[:f.find('.')] # We know the filename is BRAIN_XXXX.FACE RESPONDER etc
        if len(glob.glob('{}*'.format(stem))) == NUMBER_OF_BRAINS:
            return stem
    return []

def load(filename):
    data = pickle.load(file(filename, 'r'))
    print 'Part of brain loaded from file {} ({})'.format(filename, filesize(filename))
    return data

brain_name = find_last_valid_brain()

print '*BRAIN*', brain_name

currentSettings, wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, \
wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, \
method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, \
timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, \
durationWeight, posInSentenceWeight, \
method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, \
timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, \
durationWeight2, posInSentenceWeight2 = load('{}.{}'.format(brain_name,'ASSOCIATION'))

#r_sound_to_face, r_wordFace, r_face_to_sound, r_faceWord, \
#r_video_producer, r_wavs, r_wav_audio_ids, r_audio_classifier, \
#r_maxlen, r_NAP_hashes, r_face_id, r_face_recognizer = load('{}.{}'.format(brain_name,'RESPONDER'))


al_deleted_ids, al_NAPs, al_wavs, al_wav_audio_ids, al_NAP_hashes, al_audio_classifier, al_maxlen = load('{}.{}'.format(brain_name, 'AUDIO LEARN'))

  

