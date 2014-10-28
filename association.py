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
wordFace = {}           # {id1:[face1,numtimes],[face2,numtimes],...], id2:[...]}
faceWord ={}            # [face1:[id1,,numtimes],[id2,numtimes],[...], face2:[...]}

sentencePosition_item = [] # give each word a score for how much it belongs in the beginning of a sentence or in the end of a sentence   
wordsInSentence = {}    # list ids that has occured in the same sentence {id:[[id1,numtimes],[id2,numtimes],[id3,nt]], idN:[...]}

numWords = 4
method = 'boundedAdd'
neighborsWeight = 0.2
wordsInSentenceWeight = 0.5
similarWordsWeight = 0.5
wordFaceWeight = 0.5
faceWordWeight = 0.5
timeBeforeWeight = 0.0
timeAfterWeight = 0.6
timeDistance = 5.0
durationWeight = 0.1
posInSentenceWeight = 0.5
method2 = 'boundedAdd'
neighborsWeight2 = 0.2
wordsInSentenceWeight2 = 0.5
similarWordsWeight2 = 0.5
wordFaceWeight2 = 0.5
faceWordWeight2 = 0.5
timeBeforeWeight2 = 0.5
timeAfterWeight2 = 0.0
timeDistance2 = 5.0
durationWeight2 = 0.5
posInSentenceWeight2 = 0.5   


def association(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    assoc_in = context.socket(zmq.PULL)
    assoc_in.bind('tcp://*:{}'.format(IO.ASSOCIATION_IN))

    assoc_out = context.socket(zmq.PUSH)
    assoc_out.connect('tcp://{}:{}'.format(host, IO.ASSOCIATION_OUT))        

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    poller = zmq.Poller()
    poller.register(assoc_in, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)
    
    while True:
        #print 'assoc is running %i', time.time()
        #time.sleep(.1)
        events = dict(poller.poll())
        if assoc_in in events:
            thing = assoc_in.recv_pyobj()
            try:
                func = thing[0]
                if func == 'analyze':
                    _,wav_file,wav_segments,segment_ids,wavs,similar_ids,wordFace,faceWord = thing
                    analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,wordFace,faceWord)
                if func == 'makeSentence':
                    _,audio_id = thing
                    makeSentence(assoc_out, audio_id)                    
                if func == 'setParam':
                    _,param,value = thing
                    setParam(param,value)                    
            except Exception, e:
                print e, 'association receive failed on receiving:', thing

        if eventQ in events:
            global wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeBeforeWeight, timeAfterWeight, timeDistance, durationWeight, posInSentenceWeight, method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, timeBeforeWeight2, timeAfterWeight2, timeDistance2, durationWeight2, posInSentenceWeight2
            pushbutton = eventQ.recv_json()
            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], me.name), [ wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeBeforeWeight, timeAfterWeight, timeDistance, durationWeight, posInSentenceWeight, method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, timeBeforeWeight2, timeAfterWeight2, timeDistance2, durationWeight2, posInSentenceWeight2 ])

            if 'load' in pushbutton:
                wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeBeforeWeight, timeAfterWeight, timeDistance, durationWeight, posInSentenceWeight, method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, timeBeforeWeight2, timeAfterWeight2, timeDistance2, durationWeight2, posInSentenceWeight2 = utils.load('{}.{}'.format(pushbutton['load'], me.name))

def setParam(param,value):
    #param is a string, so we must compile the statement to set the variable
    print 'setParam:', param, value
    global numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight
    global timeBeforeWeight,timeAfterWeight,timeDistance,durationWeight,posInSentenceWeight
    global neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2
    global timeBeforeWeight2,timeAfterWeight2,timeDistance2,durationWeight2,posInSentenceWeight2   

    if param == 'numWords': numWords = int(value)
    if param == 'neighborsWeight':neighborsWeight = float(value)
    if param == 'wordsInSentenceWeight':wordsInSentenceWeight = float(value)
    if param == 'similarWordsWeight':similarWordsWeight = float(value)
    if param == 'wordFaceWeight':wordFaceWeight = float(value)
    if param == 'faceWordWeight':faceWordWeight = float(value)
    if param == 'timeBeforeWeight':timeBeforeWeight = float(value)
    if param == 'timeAfterWeight':timeAfterWeight = float(value)
    if param == 'timeDistance':timeDistance = float(value)
    if param == 'durationWeight':durationWeight = float(value)
    if param == 'posInSentenceWeight':posInSentenceWeight = float(value)
    if param == 'neighborsWeight2':neighborsWeight2 = float(value)
    if param == 'wordsInSentenceWeight2':wordsInSentenceWeight2 = float(value)
    if param == 'similarWordsWeight2':similarWordsWeight2 = float(value)
    if param == 'wordFaceWeight2':wordFaceWeight2 = float(value)
    if param == 'faceWordWeight2':faceWordWeight2 = float(value)
    if param == 'timeBeforeWeight2':timeBeforeWeight2 = float(value)
    if param == 'timeAfterWeight2':timeAfterWeight2 = float(value)
    if param == 'timeDistance2':timeDistance2 = float(value)
    if param == 'durationWeight2':durationWeight2 = float(value)
    if param == 'posInSentenceWeight2':posInSentenceWeight2 = float(value)
    if param == 'all':
        numWords = 5
        neighborsWeight = 0.0
        wordsInSentenceWeight = 0.0
        similarWordsWeight = 0.0
        wordFaceWeight = 0.0
        faceWordWeight = 0.0
        timeBeforeWeight = 0.0
        timeAfterWeight = 0.0
        timeDistance = 4.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.0
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = 0.0
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.0
        timeBeforeWeight2 = 0.0
        timeAfterWeight2 = 0.0
        timeDistance2 = 4.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0

    '''
    g = 'global '+param
    s = param+'='+float(value)
    p = 'print "'+param+' set to ",'+param
    exec(compile(g,'string','exec'))
    exec(compile(s,'string','exec'))
    exec(compile(p,'string','exec'))
    '''
    
def analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,_wordFace,_faceWord):
    
    global wordFace,faceWord
    wordFace = copy.copy(_wordFace)
    faceWord = copy.copy(_faceWord)

    markerfile = wav_file[:-4]+'.txt'
    startTime, totalDur = parseFile(markerfile) # COORDINATION! with utils.getSoundParmFromFile
    
    for i in range(len(segment_ids)):
        audio_id = segment_ids[i]
        
        # get timing and duration for segment    
        segmentStart = wav_segments[(wav_file,audio_id)][0]
        segmentDur = wav_segments[(wav_file,audio_id)][1]-segmentStart
        segmentStart += startTime
        time_word.append((segmentStart, audio_id))
        wordTime.setdefault(audio_id, []).append(segmentStart)
        duration_word.append((segmentDur, audio_id))   
        
        similar_ids_this = similar_ids[i]
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
        
def makeSentence(assoc_out, predicate):

    print 'makeSentence predicate', predicate, 'numWords', numWords, 'similarWordsWeight', similarWordsWeight

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
                            neighborsWeight, wordsInSentenceWeight, similarWordsWeight, 
                            wordFaceWeight,faceWordWeight,
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
                            neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, 
                            wordFaceWeight2,faceWordWeight2, 
                            timeBeforeWeight2, timeAfterWeight2, timeDistance2, 
                            posInSentence2, posInSentenceWidth2, posInSentenceWeight2, 
                            preferredDuration2, preferredDurationWidth2, durationWeight2)
        secondaryStream.append(secondaryAssoc)
    #print 'processing time for %i words: %f secs'%(numWords, time.time() - timeThen)
    assoc_out.send_pyobj([sentence, secondaryStream])
    
def generate(predicate, method, 
            neighborsWeight, wordsInSentenceWeight, similarWordsWeight, 
            wordFaceWeight,faceWordWeight,
            timeBeforeWeight, timeAfterWeight, timeDistance, 
            posInSentence, posInSentenceWidth, posInSentenceWeight, 
            preferredDuration, preferredDurationWidth, durationWeight):

    # get the lists we need
    #print 'get lists for predicate', predicate
    _neighbors = normalizeItemScore(copy.copy(neighbors[predicate]))
    _wordsInSentence = normalizeItemScore(copy.copy(wordsInSentence[predicate]))
    _wordFace = normalizeItemScore(copy.copy(wordFace[predicate]))
    # temporary solution to using faces
    faces = [item[0] for item in _wordFace]
    if -1 in faces: faces.remove(-1)
    face = random.choice(faces)
    print 'using face', face, 'we might want to update face/word selection'
    #print 'faceWord', faceWord
    _faceWord = normalizeItemScore(copy.copy(faceWord[face]))
    _similarWords = normalizeItemScore(copy.copy(formatAsMembership(similarWords[predicate])))
    #print '_similarWords', _similarWords
    timeContextBefore, timeContextAfter = getTimeContext(predicate, timeDistance) 
    #print '***timeContextAfter',timeContextAfter
    timeContextBefore = normalizeItemScore(timeContextBefore)
    timeContextAfter = normalizeItemScore(timeContextAfter)
    posInSentenceContext = getCandidatesFromContext(sentencePosition_item, posInSentence, posInSentenceWidth)
    durationContext = getCandidatesFromContext(duration_word, preferredDuration, preferredDurationWidth)
        
    #print 'generate lengths:', len(timeContextBefore), len(timeContextAfter), len(durationContext)
    # merge them
    if method == 'add': method = weightedSum
    if method == 'boundedAdd': method = boundedSum
    temp = method(_neighbors, neighborsWeight, _wordsInSentence, wordsInSentenceWeight)
    temp = method(temp, 1.0, _wordFace, wordFaceWeight)
    temp = method(temp, 1.0, _faceWord, faceWordWeight)
    temp = method(temp, 1.0, _similarWords, similarWordsWeight)
    temp = method(temp, 1.0, timeContextBefore, timeBeforeWeight)
    temp = method(temp, 1.0, timeContextAfter, timeAfterWeight)
    temp = method(temp, 1.0, posInSentenceContext, posInSentenceWeight)
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
        v = neighbors.setdefault(sentence[0], [])
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
        v = neighborAfter.setdefault(sentence[0], [])
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

def formatAsMembership(a):
    i = 0
    membership = []
    for item in a:
        membership.append([i,item])
    return membership

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
