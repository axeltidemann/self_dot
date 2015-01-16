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

def save(filename, data):
    pickle.dump(data, file(filename, 'w'))
    print '{} saved ({})'.format(filename, filesize(filename))

def load(filename):
    data = pickle.load(file(filename, 'r'))
    print 'Part of brain loaded from file {} ({})'.format(filename, filesize(filename))
    return data

brain_name = 'BRAIN_2015_01_15_16_55_51'# find_last_valid_brain()

print '*BRAIN*', brain_name

#face_deleted_ids, face_history, face_hashes, face_recognizer = load('{}.{}'.format(brain_name,'FACES LEARN'))

#audio_deleted_ids, NAPs, wavs, wav_audio_ids, NAP_hashes, audio_classifier, maxlen, audio_memory = load('{}.{}'.format(brain_name,'AUDIO LEARN'))

currentSettings, wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, \
wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, \
method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, \
timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, \
durationWeight, posInSentenceWeight, \
method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, \
timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, \
durationWeight2, posInSentenceWeight2 = load('{}.{}'.format(brain_name,'ASSOCIATION'))



def insertZeroMembers(l,size):
    outlist = []
    for i in range(size):
        outlist.append([i,0])
    for item in l:
        outlist[item[0]] = item
    return outlist
    

def make_scorelist(context):
    '''take a list of [[id,score],[id,score]...], return a list of scores'''
    outlist = []
    for item in context:
        outlist.append(item[1])
    return outlist
    


            


