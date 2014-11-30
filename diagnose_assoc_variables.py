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

def remove_duplicates_tuple(l):
    l = list(set(l))
    for i in range(len(l)):
        l[i] = list(l[i])
    for item in l: item.reverse()
    l.sort()
    for item in l: item.reverse()
    for i in range(len(l)):
        l[i] = tuple(l[i])
    return l
    
def remove_duplicates_dict(d):
    for k in d.iterkeys():
        d[k] = list(set(d[k]))
    return d
    
duration_word = remove_duplicates_tuple(duration_word)
wordTime = remove_duplicates_dict(wordTime)
#save('{}.{}'.format(brain_name,'ASSOCIATION'))

def insertZeroMembers(l,size):
    outlist = []
    for i in range(size):
        outlist.append([i,0])
    for item in l:
        outlist[item[0]] = item
    return outlist
    
# inspiser de lister hvor du har flere kandidater med samme audio_id men forskjellig membership
# se om du kan beholde kun den kandidaten som har høyest membership
# insertZeroMembers
# forenkle weightedSum og boundedSum til ren array arithmetic
# alle lister har membership_format [[id1,score],[id2,score],[id3,sc]] etter prosessering i generate

## sjekk at wordsInSentence har format [id,numtimes] og ingen duplicates

## get...Context kan ha duplicates, "hei du er dum du" med predicate "hei", [du,1] [er,0.8] [dum,0.6] [du,0.4] ...
# func for remove_duplicates_from_context (select='highest')
# for each: if item[0] in otheritem[0] and item[1] < otheritem[1]: remove item

testContext = [[1,0.9],[2,0.2],[3,0.7],[1,0.6],[2,0.8],[1,0.1],[3,0.1],[2,0.5]]

def remove_duplicates_from_context(context):
    remove_list = []
    for item in context:
        print 'item', item
        without_me = copy.copy(context)
        without_me.remove(item)
        print 'without_me', without_me
        for otheritem in without_me:
            print 'test', item, otheritem
            if item[0] == otheritem[0] and item[1] < otheritem[1]:
                print 'remove', item
                remove_list.append(item)
                break
    for duplicate in remove_list:
        context.remove(duplicate)


            

