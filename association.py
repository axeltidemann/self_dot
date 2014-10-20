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


import multiprocessing as mp
import zmq
import IO
import re
findfloat=re.compile(r"[0-9.]*")
import copy
import random
import numpy
import math
import utils
import time

#wavSegments = {}        # {(wavefile, id):[segstart,segend], (wavefile, id):[segstart,segend], ...} 
#wavs_as_words = []      # list of wave files for each word (audio id) [[w1,w2,w3],[w4],...]
wordTime = {}           # {id1:[time1, time2, t3...]], id2: ...}
time_word = []          # [[time,id1],[time,id2],...]
duration_word = []      # [[dur, id1], [dur,id2]
similarWords = {}       # {id1: [distance to audio id 0, distance to audio id 1, ...], id2: [distance to audio id 0, distance to audio id 1, ...]
neighbors = {}          # {id1: [[neighb_id1, how_many_times], [neighb_id2, how_many_times]...], id2:[[n,h],[n,h]...]}
neighborAfter = {}      # as above, but only including words that immediately follow this word (not immediately preceding)
wordFace = {}           # {id1:[face1,face2,...], id2:[...]}
faceWord ={}            # [face1:[id1,id2...], face2:[...]}

sentencePosition_item = [] # give each word a score for how much it belongs in the beginning of a sentence or in the end of a sentence   
wordsInSentence = {}    # list ids that has occured in the same sentence {id:[[id1,numtimes],[id2,numtimes],[id3,nt]], idN:[...]}



def association(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    assoc_in = context.socket(zmq.PULL)
    assoc_in.bind('tcp://*:{}'.format(IO.ASSOCIATION_IN))

    assoc_out = context.socket(zmq.PUSH)
    assoc_out.connect('tcp://{}:{}'.format(host, IO.ASSOCIATION_OUT))        

    poller = zmq.Poller()
    poller.register(assoc_in, zmq.POLLIN)
    
    while True:
        #print 'assoc is running %i', time.time()
        #time.sleep(.1)
        events = dict(poller.poll())
        if assoc_in in events:
            thing = assoc_in.recv_pyobj()
            try:
                func = thing[0]
                if func == 'analyze':
                    _,wav_file,wav_segments,segment_ids,wavs,similar_ids,sound_to_face,face_to_sound = thing
                    analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,sound_to_face,face_to_sound)
                if func == 'makeSentence':
                    dummy,audio_id,numWords,method,timeBeforeWeight,timeAfterWeight,timeDistance,\
                    durationWeight,posInSentenceWeight,method2,timeBeforeWeight2,timeAfterWeight2,\
                    timeDistance2,durationWeight2,posInSentenceWeight2 = thing
                    makeSentence(assoc_out,audio_id,numWords,method,timeBeforeWeight,timeAfterWeight,timeDistance,\
                    durationWeight,posInSentenceWeight,method2,timeBeforeWeight2,timeAfterWeight2,\
                    timeDistance2,durationWeight2,posInSentenceWeight2)
            except Exception, e:
                print e, 'association receive failed on receiving:', thing

def analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,sound_to_face,face_to_sound):
    #print '*** *** assoc analyze:'
    #print 'wav_file',wav_file
    #print 'wav_segments',wav_segments
    #print 'segment_ids',segment_ids
    #print 'wavs',wavs
    #print 'similar_ids',similar_ids
    #print 'sound_to_face',sound_to_face
    #print 'face_to_sound', face_to_sound
    
    global wordFace,faceWord
    wordFace = copy.copy(sound_to_face)
    faceWord = copy.copy(face_to_sound)

    markerfile = wav_file[:-4]+'.txt'
    startTime, totalDur = parseFile(markerfile) # COORDINATION! with utils.getSoundParmFromFile
    
    for i in range(len(segment_ids)):
        audio_id = segment_ids[i]
        
        # get timing and duration for segment    
        segmentStart = wav_segments[(wav_file,audio_id)][0]
        segmentDur = wav_segments[(wav_file,audio_id)][1]-segmentStart
        print '**** segmentStart', segmentStart, startTime
        segmentStart += startTime
        time_word.append((segmentStart, audio_id))
        wordTime.setdefault(audio_id, []).append(segmentStart)
        duration_word.append((segmentDur, audio_id))   
        
        similar_ids_this = similar_ids[i]
        if max(similar_ids_this) == 0:
            similarScaler = 1
        else:
            similarScaler = 1/float(max(similar_ids_this))
        similarWords[audio_id] = scale(similar_ids_this, similarScaler)
    
    # analysis of the segment's relationship to the sentence it occured in
    updateWordsInSentence(segment_ids)
    updateNeighbors(segment_ids)
    updateNeighborAfter(segment_ids)
    updatePositionMembership(segment_ids) 
    #print '** segment_ids', segment_ids
    #print '** wordsInSentence', wordsInSentence
    #for item in time_word:
    #    print '* time_word', item
    #print 'wordTime', wordTime
    #for item in duration_word:
    #    print '* duration_word', item
    
    #posInSentenceContext = getCandidatesFromContext(l.sentencePosition_item, posInSentence, posInSentenceWidth)
    
def makeSentence(assoc_out, predicate, numWords, method,
                timeBeforeWeight, timeAfterWeight, timeDistance, 
                durationWeight, posInSentenceWeight,
                method2, 
                timeBeforeWeight2, timeAfterWeight2, timeDistance2, 
                durationWeight2, posInSentenceWeight2):

    print 'makeSentence predicate', predicate
    #print 'stuff we need:'
    #print  wavs_as_words, wavSegments, time_word, wordTime, duration_word, similarWords, wordFace, faceWord
    #print '***'
    
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
    #print 'sentence', sentence
    #print 'secondaryStream', secondaryStream
    #print 'processing time for %i words: %f secs'%(numWords, time.time() - timeThen)
    #return sentence, secondaryStream
    assoc_out.send_pyobj([sentence, secondaryStream])
    
def generate(predicate, method, 
            timeBeforeWeight, timeAfterWeight, timeDistance, 
            posInSentence, posInSentenceWidth, posInSentenceWeight, 
            preferredDuration, preferredDurationWidth, durationWeight):
    # get the lists we need
    #print 'get lists for predicate', predicate
    timeContextBefore, timeContextAfter = getTimeContext(predicate, timeDistance) 
    timeContextBefore = normalizeItemScore(timeContextBefore)
    timeContextAfter = normalizeItemScore(timeContextAfter)
    durationContext = getCandidatesFromContext(duration_word, preferredDuration, preferredDurationWidth)
    #print 'generate lengths:', len(timeContextBefore), len(timeContextAfter), len(durationContext)
    # merge them
    if method == 'add': method = weightedSum
    if method == 'boundedAdd': method = boundedSum
    temp = method(timeContextBefore, timeBeforeWeight, timeContextAfter, timeAfterWeight)
    temp = method(temp, 1.0, durationContext, durationWeight) 
    #print 'generate temp', temp       
    # select the one with the highest score
    if len(temp) < 1:
        nextWord = predicate
        print '** WARNING: ASSOCIATION GENERATE HAS NO VALID ASSOCIATIONS, RETURNING PREDICATE'
    else:
        nextWord = select(temp, 'highest')
    return nextWord
    
def parseFile(markerfile):
    f = open(markerfile, 'r')
    #segments = []
    enable = 0
    startTime = 0
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        startTime = float(line[30:])
        if 'Total duration:'  in line: 
            enable = 0
            totalDur = float(line[16:])
        #if enable:
        #    segments.append(float(line)) 
        if 'Sub segment start times:' in line: enable = 1
    return startTime, totalDur

def updateWordsInSentence(sentence):
    a = 0
    for word in sentence:
        v = wordsInSentence.setdefault(word, [])
        b = 0
        for other in sentence:
            if a != b: #don't count itself as an occurence
                otherInSentence = 0
                for word_score in v:
                    if other == word_score[0]:
                        word_score[1] += 1
                        otherInSentence = 1
                if otherInSentence == 0:
                    v.append([other, 1])
            b += 1
        a += 1

def updateNeighbors(sentence):    
    if len(sentence) == 1:
        return
    for i in range(len(sentence)):
        v = neighbors.setdefault(sentence[i], [])
        if i == 0: neighborPos = [i+1]
        elif i == len(sentence)-1: neighborPos = [i-1] 
        else: neighborPos = [i-1,i+1]
        for n in neighborPos:
            alreadyHere = 0
            for word_score in v:
                if sentence[n] == word_score[0]:
                    word_score[1] += 1
                    alreadyHere = 1
            if alreadyHere == 0:
                v.append([sentence[n], 1])

def updateNeighborAfter(sentence): 
    if len(sentence) == 1:
        return
    for i in range(len(sentence)-1):
        v = neighborAfter.setdefault(sentence[i], [])
        if sentence[i+1] in [v[j][0] for j in range(len(v))]:
            for word_score in v:
                if sentence[i+1] == word_score[0]:
                    word_score[1] += 1
        else:
            v.append([sentence[i+1], 1])

# give each word a score for how much it belongs in the beginning of a sentence or in the end of a sentence   
# The resulting list here just gives the position (= the score for belonging in the end of a sentence).
# The score for membership in the beginning of a sentence is simply (1-endscore)
def updatePositionMembership(sentence):
    global sentencePosition_item
    lenSent = len(sentence)
    for i in range(lenSent):
        if lenSent == 1: position = 1.0
        else: position = int((i/float(lenSent-1))*10)/10.0
        sentencePosition_item.append((position,sentence[i]))
    sentencePosition_item = list(set(sentencePosition_item))
    sentencePosition_item.sort()


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
    #print 'getTimeContext wordTime', wordTime
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
    #print 'time OK', invertedTimeContextBefore, invertedTimeContextAfter
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
    print 'run as main'
