#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
A word is a classified segment (audio clip), as part of a sentence (audio file).
Classification of segments can be ambigous and can be updated/changed at a later time (dream state).

The audio_id in brain.wavs corresponds exactly to the semantic unit 'word'

allWords = list of all unique words (after classification)
wordTime = time when each word was recorded
timeWord = word recorded at a specific time
durationWord = duration of word
similarWords = words that sound similar but are not classified as the same
neighbors = words used immediately before or after this word
neighborsAfter = words used immediately after this word
wordFace = who has said this word (may be several faces)
faceWord = all words said by this face
'''


import re 
import re
findfloat=re.compile(r"[0-9.]*")
import copy
import random
import numpy
import math

wavs_as_words = []      # list of wave files for each word (audio id) [[w1,w2,w3],[w4],...]
wordTime = {}           # {id1:[time1, time2, t3...]], id2: ...}
time_word = []          # [[time,id1],[time,id2],...]
duration_word = []      # [[dur, id1], [dur,id2]
similarWords = {}       # {id1: [[similar_id1, distance], [sim_id2, d], [sim_id3,d]], id2: [...]}
neighbors = {}          # {id1: [[neighb_id1, how_many_times], [neighb_id2, how_many_times]...], id2:[[n,h],[n,h]...]}
neighborAfter = {}      # as above, but only including words that immediately follow this word (not immediately preceding)
wordFace = {}           # {id1:[face1,face2,...], id2:[...]}
faceWord ={}            # [face1:[id1,id2...], face2:[...]}

def analyze(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound):
    #print_us(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound)

    markerfile = filename[:-4]+'.txt'
    startTime, totalDur, segments = parseFile(markerfile)

    wavs_as_words = copy.copy(wavs)
    time_word.append((startTime, audio_id))
    wordTime.setdefault(audio_id, []).append(startTime)
    duration_word.append((totalDur, audio_id))
    #similarWords =      
    #neighbors = 
    #neighborAfter = 
    wordFace = copy.copy(sound_to_face)
    faceWord = copy.copy(face_to_sound)
    
    
def parseFile(markerfile):
    f = open(markerfile, 'r')
    segments = []
    enable = 0
    startTime = 0
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        startTime = float(line[30:])
        if 'Total duration:'  in line: 
            enable = 0
            totalDur = float(line[16:])
        if enable:
            segments.append(float(line)+startTime) 
        if 'Sub segment start times:' in line: enable = 1
    return startTime, totalDur, segments


def print_us(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound):
    print '*** association analysis ***'
    print filename

    print '\n***' 
    print audio_id

    print '\n***' 
    print wavs
    
    print '\n***' 
    # would like to have distance from id to all other ids not in same class
    # or distance from this class to all other classes 
    print audio_hammings # distance from id to all other in same class?

    print '\n***' 
    print sound_to_face

    print '\n***' 
    print face_to_sound
    print '\n***' 
    
def makeSentence(predicate, numWords, method,
                timeBeforeWeight, timeAfterWeight, timeDistance, 
                durationWeight,
                method2, 
                timeBeforeWeight2, timeAfterWeight2, timeDistance2, 
                durationWeight2):

    print 'makeSentence predicate', predicate
    sentence = [predicate]
    secondaryStream = []
    #timeThen = time.time()
    for i in range(numWords):
        posInSentence = i/float(numWords-1)
        posInSentenceWidth = 0.2
        if posInSentence < 0.5:
            preferredDuration = posInSentence*20 # longer words in the middle of a sentence (just as a test...)
        else:
            preferredDuration = (1-posInSentence)*20
        preferredDurationWidth = 3
        prevPredicate = predicate # save it for the secondary association
        predicate = generate(predicate, method, 
                            timeBeforeWeight, timeAfterWeight, timeDistance, 
                            posInSentence, posInSentenceWidth, posInSentenceWeight, 
                            preferredDuration, preferredDurationWidth, durationWeight)
        sentence.append(predicate)
        # secondary association for the same predicate
        posInSentence2 = posInSentence
        posInSentenceWidth2 = posInSentenceWidth
        preferredDuration2 = preferredDuration*3
        preferredDurationWidth2 = preferredDurationWidth
        secondaryAssoc = generate(prevPredicate, method2, 
                            timeBeforeWeight2, timeAfterWeight2, timeDistance2, 
                            posInSentence2, posInSentenceWidth2, posInSentenceWeight2, 
                            preferredDuration2, preferredDurationWidth2, durationWeight2)
        secondaryStream.append(secondaryAssoc)
    print 'sentence', sentence
    print 'secondaryStream', secondaryStream
    #print 'processing time for %i words: %f secs'%(numWords, time.time() - timeThen)

def generate(predicate, method, 
            timeBeforeWeight, timeAfterWeight, timeDistance, 
            posInSentence, posInSentenceWidth, posInSentenceWeight, 
            preferredDuration, preferredDurationWidth, durationWeight):
    # get the lists we need
    timeContextBefore, timeContextAfter = getTimeContext(predicate, timeDistance) 
    timeContextBefore = normalizeItemScore(timeContextBefore)
    timeContextAfter = normalizeItemScore(timeContextAfter)
    durationContext = getCandidatesFromContext(duration_item, preferredDuration, preferredDurationWidth)
    # merge them
    if method == 'add': method = weightedSum
    if method == 'boundedAdd': method = boundedSum
    temp = method(timeContextBefore, timeBeforeWeight, timeContextAfter, timeAfterWeight)
    temp = method(temp, 1.0, durationContext, durationWeight)        
    # select the one with the highest score
    nextWord = select(temp, 'highest')
    return nextWord

def getTimeContext(predicate, distance):
    '''
    * Look up the time(s) when predicate has been used (wordTime),
    * Look up these times (time_word), generating a list of words
    appearing within a desired distance in time from each of these time points.
    * Make a new list of [word, distance], with the distance inverted (use maxDistance - distance) and normalized.
    Invert items [distance,word] and sort, invert items again.
    This list will have items (words) sorted from close in time to far in time, retaining a normalized score 
    for how far in time from the predicate each word has occured.
    '''
    timeWhenUsed = wordTime[predicate]
    quantize = 0.01
    iquantize = 1/quantize
    timeContextBefore = []
    timeContextAfter = []
    for t in timeWhenUsed:
        startIndex = -1
        endIndex = -1
        for i in range(len(time_word)):
            if (time_word[i][0] > (t-distance)) and (startIndex == -1):
                startIndex = i
            if (time_word[i][0] > (t+distance)):
                endIndex = i
                break
        if startIndex == -1: startIndex = 0
        if endIndex == -1: endIndex = len(time_word)
        for j in range(startIndex, endIndex):
            if time_word[j][0]-t > 0 : # do not include the query word
                timeContextAfter.append(((int(time_word[j][0]*iquantize)-int(t*iquantize)),time_word[j][1]))
            if time_word[j][0]-t < 0 : # do not include the query word
                timeContextBefore.append((int(t*iquantize)-(int(time_word[j][0]*iquantize)),time_word[j][1]))
    if len(timeContextBefore) > 0:
        s_timeContextBefore = set(timeContextBefore)
        l_timeContextBefore = list(s_timeContextBefore)
        l_timeContextBefore.sort()
        invertedTimeContextBefore = []
        for item in l_timeContextBefore:
            invTime = (distance-item[0]*quantize)
            invertedTimeContextBefore.append([item[1], invTime])
    else: invertedTimeContextBefore = []
    if len(timeContextAfter) > 0:
        s_timeContextAfter = set(timeContextAfter)
        l_timeContextAfter = list(s_timeContextAfter)
        l_timeContextAfter.sort()
        invertedTimeContextAfter = []
        for item in l_timeContextAfter:
            invTime = (distance-item[0]*quantize)
            invertedTimeContextAfter.append([item[1], invTime])
    else: invertedTimeContextAfter = []
    return invertedTimeContextBefore, invertedTimeContextAfter

def getCandidatesFromContext(context, position, width):
    candidates = []
    for item in context:
        if item[0] < position-width:
            continue
        if item[0] <= position+width:
            membership = 1-(abs(position-item[0])/float(width))
            candidates.append([item[1], membership])
        else:
            break
    return candidates

def weightedSum(a_, weightA_, b_, weightB_):
    '''
    Add the score of items found in both sets,
    use score as is for items found in only one of the sets.
    '''
    c = []
    if len(a_) > len(b_):
        a = copy.copy(a_)
        weightA = weightA_
        b = copy.copy(b_)
        weightB = weightB_
    else:
        a = copy.copy(b_)
        weightA = weightB_
        b = copy.copy(a_)
        weightB = weightA_        
    itemsA = [a[i][0] for i in range(len(a))]
    itemsB = [b[i][0] for i in range(len(b))]
    scoreA = scale([float(a[i][1]) for i in range(len(a))],weightA)
    scoreB = scale([float(b[i][1]) for i in range(len(b))],weightB)
    #print 'weightedSum', min(scoreA), max(scoreA), min(scoreB), max(scoreB)
    removeFromB = set()
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        a_in_b = 0
        itemsBB = copy.copy(itemsB)
        scoreBB = copy.copy(scoreB)
        while itemA in itemsBB:
            a_in_b = 1
            j = itemsBB.index(itemA)
            c.append([itemsA[i], scoreBB[j] + scoreA[i]])    # add scores and put item in c
            itemsBB[j] = '_'      # hide used elements but keep indexing unmodified
            scoreBB[j] = '_'
            removeFromB.add(j)   # mark this one to be removed later
        if not a_in_b:
            c.append([itemsA[i], scoreA[i]])         
    removeFromB = list(removeFromB)
    removeFromB.sort()  
    removeFromB.reverse() # ... so we can use pop(n), starting from the end
    for i in removeFromB:
        gone = b.pop(i)
    for i in range(len(b)):
        c.append([b[i][0], b[i][1]*weightB])
    return c
          
def boundedSum(a_, weightA_, b_, weightB_):
    '''
    Bounded sum, scores are clipped to the "weight" value.
    Add the score of items found in both sets,
    use score as is for items found in only one of the sets.
    '''
    c = []
    if len(a_) > len(b_):
        a = copy.copy(a_)
        weightA = weightA_
        b = copy.copy(b_)
        weightB = weightB_
    else:
        a = copy.copy(b_)
        weightA = weightB_
        b = copy.copy(a_)
        weightB = weightA_        
    itemsA = [a[i][0] for i in range(len(a))]
    itemsB = [b[i][0] for i in range(len(b))]
    scoreA = clip([float(a[i][1]) for i in range(len(a))],weightA)
    scoreB = clip([float(b[i][1]) for i in range(len(b))],weightB)
    #print 'boundedSum', min(scoreA), max(scoreA), min(scoreB), max(scoreB)
    removeFromB = set()
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        a_in_b = 0
        itemsBB = copy.copy(itemsB)
        scoreBB = copy.copy(scoreB)
        while itemA in itemsBB:
            a_in_b = 1
            j = itemsBB.index(itemA)
            c.append([itemsA[i], scoreBB[j] + scoreA[i]])    # add scores and put item in c
            itemsBB[j] = '_'      # hide used elements but keep indexing unmodified
            scoreBB[j] = '_'
            removeFromB.add(j)   # mark this one to be removed later
        if not a_in_b:
            c.append([itemsA[i], scoreA[i]])         
    removeFromB = list(removeFromB)
    removeFromB.sort()  
    removeFromB.reverse() # ... so we can use pop(n), starting from the end
    for i in removeFromB:
        gone = b.pop(i)
    for i in range(len(b)):
            bVal = b[i][1]
            if bVal > weightB: bVal = weightB
            c.append([b[i][0], bVal])
    return c

def normalizeItemScore(a):
    if a != []:
        highest = float(max([a[i][1] for i in range(len(a))]))
        if highest != 0:
            for i in range(len(a)):
                a[i][1] /= highest
    return a

def scale(a, scale):
    a = list(numpy.array(a)*scale)
    return a

def clip(a, clipVal):
    if clipVal >= 0:
        a = list(numpy.clip(a, 0, clipVal))
    else:
        a = list(numpy.array(numpy.clip(a, 0, abs(clipVal)))*-1)
    return a

def select(items, method):
    words = [items[i][0] for i in range(len(items))]
    scores = [items[i][1] for i in range(len(items))]
    #print '\n **debug select; len, min, max:', len(words),min(scores), max(scores) 
    #print 'temp', items
    if method == 'highest':
        return words[scores.index(max(scores))]
    elif method == 'lowest':
        return words[scores.index(min(scores))]
    else:
        random.choice(words)

if __name__ == '__main__':
    analyze()
