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
import numpy as np
import random
import math
import utils
import time
import cPickle as pickle

from pyevolve import G1DList, GSimpleGA, Mutators, Selectors, Initializators, Mutators

#wavSegments = {}        # {(wavefile, id):[segstart,segend], (wavefile, id):[segstart,segend], ...} 
#wavs_as_words = []      # list of wave files for each word (audio id) [[w1,w2,w3],[w4],...]
#wavfile_words = {}      # {wavefile1:[audio_id1, id2, id3], wavefile2:[audio_id1, id2, id3],...}
wordTime = {}           # {id1:[time1, time2, t3...]], id2: ...}
time_word = []          # [[time,id1],[time,id2],...]
duration_word = []      # [[dur, id1], [dur,id2]
similarWords = {}       # {id1: [distance to audio id 0, distance to audio id 1, ...], id2: [distance to audio id 0, distance to audio id 1, ...]
neighbors = {}          # {id1: [[neighb_id1, how_many_times], [neighb_id2, how_many_times]...], id2:[[n,h],[n,h]...]}
neighborAfter = {}      # as above, but only including words that immediately follow this word (not immediately preceding)
wordFace = {}           # {id1:[face1,numtimes],[face2,numtimes],...], id2:[...]}
faceWord ={}            # [face1:[id1,numtimes],[id2,numtimes],[...], face2:[...]}

sentencePosition_item = [] # give each word a score for how much it belongs in the beginning of a sentence or in the end of a sentence   
wordsInSentence = {}    # list ids that has occured in the same sentence {id:[[id1,numtimes],[id2,numtimes],[id3,nt]], idN:[...]}

numWords = 4
method = 'boundedAdd'
neighborsWeight = 0.0
wordsInSentenceWeight = 0.3
similarWordsWeight = 0.2
wordFaceWeight = 0.0    
faceWordWeight = 0.1
timeShortBeforeWeight = 0.3
timeShortAfterWeight = 0.4
timeShortDistance = 12.0
timeLongBeforeWeight = 0.02
timeLongAfterWeight = 0.05
timeLongDistance = 120.0
durationWeight = 0.0
posInSentenceWeight = 0.5
method2 = 'boundedAdd'
neighborsWeight2 = 0.0
wordsInSentenceWeight2 = 0.1
similarWordsWeight2 = 0.3
wordFaceWeight2 = 0.0
faceWordWeight2 = 0.0
timeShortBeforeWeight2 = 0.1
timeShortAfterWeight2 = 0.4
timeShortDistance2 = 18.0
timeLongBeforeWeight2 = 0.02
timeLongAfterWeight2 = 0.05
timeLongDistance2 = 120.0
durationWeight2 = 0.0
posInSentenceWeight2 = 0.2  

LastSentenceID_debug = 0

currentSettings = [] # for temporal storage of globals (association weights)

def association(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    association = context.socket(zmq.ROUTER)
    association.bind('tcp://*:{}'.format(IO.ASSOCIATION))

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, IO.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    poller = zmq.Poller()
    poller.register(association, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)
    
    while True:
        #print 'assoc is running %i', time.time()
        #time.sleep(.1)
        events = dict(poller.poll())
        if association in events:
            address, _, message = association.recv_multipart()
            thing = pickle.loads(message)
            try:
                func = thing[0]
                answer = 'Done'
                if func == 'analyze':
                    _,wav_file,wav_segments,segment_ids,wavs,similar_ids,wordFace,faceWord = thing
                    analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,wordFace,faceWord)
                if func == 'makeSentence':
                    _,audio_id = thing
                    answer = makeSentence(audio_id)                    
                if func == 'setParam':
                    _,param,value = thing
                    setParam(param,value)                    
                if func == 'evolve':
                    evolve_sentence_parameters()
                if func == 'getSimilarWords':
                    _,predicate, distance = thing
                    answer = getSimilarWords(predicate, distance)
                if func == 'getFaceResponse':
                    _,face, = thing
                    answer = getFaceResponse(face)
                if func == 'pushCurrentSettings':
                    pushCurrentSettings()
                if func == 'popCurrentSettings':
                    popCurrentSettings()

                association.send_multipart([ address,
                                             b'',
                                             pickle.dumps(answer) ])

            except Exception, e:
                print utils.print_exception('association receive failed on receiving {}'.format(thing))
                association.send_multipart([ address,
                                             b'',
                                             pickle.dumps('Something went wrong in association.py, check the output log!') ])


        if eventQ in events:
            global wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, \
                   wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords,\
                   method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, \
                   timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, \
                   durationWeight, posInSentenceWeight, \
                   method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, \
                   timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, \
                   durationWeight2, posInSentenceWeight2
            pushbutton = eventQ.recv_json()
            if 'save' in pushbutton:
                utils.save('{}.{}'.format(pushbutton['save'], me.name), [ wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, \
                        wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, \
                        method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, \
                        timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, \
                        durationWeight, posInSentenceWeight, \
                        method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, \
                        timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, \
                        durationWeight2, posInSentenceWeight2 ])

            if 'load' in pushbutton:
                wordTime, time_word, duration_word, similarWords, neighbors, neighborAfter, \
                wordFace, faceWord, sentencePosition_item, wordsInSentence, numWords, \
                method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, \
                timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, \
                durationWeight, posInSentenceWeight, \
                method2, neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, wordFaceWeight2, faceWordWeight2, \
                timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, \
                durationWeight2, posInSentenceWeight2 = utils.load('{}.{}'.format(pushbutton['load'], me.name))

def analyze(wav_file,wav_segments,segment_ids,wavs,similar_ids,_wordFace,_faceWord):
    print 'analyze', wav_file, segment_ids
    global LastSentenceID_debug
    if segment_ids[0]+1 < LastSentenceID_debug:
        print '*********\n\n*********\n\n*********\n\n*********\n\n**Something DID go WRONG with the IDs !!!!!'
        print '*********\n\n*********\n\n*********\n\n*********\n\n'
    #print 'wordTime',wordTime
    #print 'similar_ids:'
    #for item in similar_ids:
    #    print '   ', item
    global wordFace,faceWord#,wavfile_words
    wordFace = copy.copy(_wordFace)
    faceWord = copy.copy(_faceWord)
    #wavfile_words[wav_file]= segment_ids

    markerfile = wav_file[:-4]+'.txt'
    startTime,totalDur,_,_ = utils.getSoundInfo(markerfile)
    
    for i in range(len(segment_ids)):
        audio_id = segment_ids[i]
        
        # get timing and duration for segment    
        segmentStart = wav_segments[(wav_file,audio_id)][0]
        segmentDur = int((wav_segments[(wav_file,audio_id)][1]-segmentStart)*10)/10.0
        
        segmentStart += startTime
        time_word.append((segmentStart, audio_id))
        wordTime.setdefault(audio_id, []).append(segmentStart)
        duration_word.append((segmentDur, audio_id))   
        
        wordTime.setdefault(audio_id, []).append(segmentStart)
        duration_word.append((segmentDur, audio_id))   
        
        similar_ids_this = similar_ids[i]
        #if max(similar_ids_this) == 0:
        #    similarScaler = 1
        #else:
        #    similarScaler = 1/float(max(similar_ids_this))
        #similarWords[audio_id] = scale(similar_ids_this, similarScaler)
        similarWords[audio_id] = similar_ids_this
        
        # fill in our best estimates for hamming distances between this id and earlier ids
        # (these will be updated with better values when a new sound similar to any earlier id comes in) 
        for k in similarWords.keys():
            #print 'checking', similarWords[k], similarWords[audio_id], 'length k, a',len(similarWords[k]), len(similarWords[audio_id])
            if len(similarWords[k]) < len(similarWords[audio_id]):
                #print 'adding id %i, distance %f to %i'%(audio_id,similarWords[audio_id][k], k)
                similarWords[k].append(similarWords[audio_id][k])
    
    # analysis of the segment's relationship to the sentence it occured in
    print 'update segments in sentence...'
    updateWordsInSentence(segment_ids)
    updateNeighbors(segment_ids)
    updateNeighborAfter(segment_ids)
    updatePositionMembership(segment_ids) 
    print '...analysis done'
        
def makeSentence(predicate):

    print 'makeSentence predicate', predicate, 'numWords', numWords, 'similarWordsWeight', similarWordsWeight

    sentence = [predicate]
    secondaryStream = []
    
    # secondary association for predicate
    posInSentence2 = 0
    posInSentenceWidth2 = 1
    preferredDuration2 = 2
    preferredDurationWidth2 = 1
    secondaryAssoc = generate(predicate, method2,
                        neighborsWeight2, wordsInSentenceWeight2, similarWordsWeight2, 
                        wordFaceWeight2,faceWordWeight2, 
                        timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, 
                        timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, 
                        posInSentence2, posInSentenceWidth2, posInSentenceWeight2, 
                        preferredDuration2, preferredDurationWidth2, durationWeight2)
    secondaryStream.append(secondaryAssoc)
    
    #timeThen = time.time()
    for i in range(numWords):
        posInSentence = i/float(numWords)
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
                            timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, 
                            timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, 
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
                            timeShortBeforeWeight2, timeShortAfterWeight2, timeShortDistance2, 
                            timeLongBeforeWeight2, timeLongAfterWeight2, timeLongDistance2, 
                            posInSentence2, posInSentenceWidth2, posInSentenceWeight2, 
                            preferredDuration2, preferredDurationWidth2, durationWeight2)
        secondaryStream.append(secondaryAssoc)
        
    #print 'processing time for %i words: %f secs'%(numWords, time.time() - timeThen)
    return [sentence, secondaryStream]

def sentence_fitness(genome):
    debug = False
    global neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeShortBeforeWeight, timeShortAfterWeight, timeLongBeforeWeight, timeLongAfterWeight, posInSentenceWeight
    neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeShortBeforeWeight, timeShortAfterWeight, timeLongBeforeWeight, timeLongAfterWeight, posInSentenceWeight = genome
    fitness = 0
    for predicate in [10]:#range(len(wordTime.keys())):
        if debug: print 'PREDICATE', predicate, '/', len(wordTime.keys())
        sentence = simple_make_sentence(predicate)

        neighbors_to_this = [ item[0] for item in neighbors[predicate] if item[1] ]
        if debug: print 'neighbors_to_this', neighbors_to_this
        fitness += np.mean([ word in neighbors_to_this for word in sentence ])

        in_sentence = [ item[0] for item in wordsInSentence[predicate] if item[1] ]
        if debug: print 'in_sentence', in_sentence
        fitness += np.mean([ word in in_sentence for word in sentence ])

        faces = [item[0] for item in wordFace[predicate]]
        if debug: print 'faces', faces
        if -1 in faces: faces.remove(-1)
        if faces == []: face = 0
        else: face = random.choice(faces)

        try:
            face_word =  [ item[0] for item in faceWord[face] if item[1] ]
            if debug: print 'face_word', face_word
            fitness += np.mean([ word in face_word for word in sentence ])
        except:
            pass

        similar_scores = similarWords[predicate]
        idxs = np.argsort(similar_scores)
        winners = idxs[-int(len(idxs)/3):]
        if debug: print 'winners', winners
        fitness += np.mean([ word in winners for word in sentence ])

        time_short_before, time_short_after = getTimeContext(predicate, timeShortDistance)
        time_short_before = [ item[0] for item in time_short_before ]
        time_short_after = [ item[0] for item in time_short_after ]
        if debug: print 'time_short_before', time_short_before
        if debug: print 'time_short_after', time_short_after
        fitness += np.mean([ word in time_short_before for word in sentence ])
        fitness += np.mean([ word in time_short_after for word in sentence ])

        time_long_before, time_long_after = getTimeContext(predicate, timeLongDistance)
        time_long_before = [ item[0] for item in time_long_before ]
        time_long_after = [ item[0] for item in time_long_after ]

        if debug: print 'time_long_before', time_long_before
        if debug: print 'time_long_after', time_long_after
        fitness += np.mean([ word in time_long_before for word in sentence ])
        fitness += np.mean([ word in time_long_after for word in sentence ])

        # HERE BE DRAGONS!
        posInSentenceWidth = .2     
        posInSentence = .5
        pos_in_sentence_context = getCandidatesFromContext(sentencePosition_item, posInSentence, posInSentenceWidth)
        pos_in_sentence_context = [ item[0] for item in pos_in_sentence_context ]
        if debug: print 'pos_in_sentence_context', pos_in_sentence_context 
        fitness += np.mean([ word in pos_in_sentence_context for word in sentence ])    

    return fitness
    
def evolve_sentence_parameters():
    print 'Using artificial evolution to find associations weights'
    genome = G1DList.G1DList(10)
    genome.evaluator.set(sentence_fitness)
    genome.setParams(rangemin=0.0, rangemax=1.0)   
    genome.initializator.set(Initializators.G1DListInitializatorReal)
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setMutationRate(.5)
    ga.setPopulationSize(100)
    ga.setGenerations(100)
    ga.setElitism(True)
    ga.setElitismReplacement(1)
    ga.evolve(freq_stats=10)
    print ga.bestIndividual()
    global neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeShortBeforeWeight, timeShortAfterWeight, timeLongBeforeWeight, timeLongAfterWeight, posInSentenceWeight
    neighborsWeight, wordsInSentenceWeight, similarWordsWeight, wordFaceWeight, faceWordWeight, timeShortBeforeWeight, timeShortAfterWeight, timeLongBeforeWeight, timeLongAfterWeight, posInSentenceWeight = ga.bestIndividual()
    print 'Weights updated after evolution'

def simple_make_sentence(predicate):
    sentence = [predicate]
    for i in range(numWords):
        posInSentence = i/float(numWords)
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
                            timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, 
                            timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, 
                            posInSentence, posInSentenceWidth, posInSentenceWeight, 
                            preferredDuration, preferredDurationWidth, durationWeight)
        sentence.append(predicate)
    return sentence
    
def generate(predicate, method, 
            neighborsWeight, wordsInSentenceWeight, similarWordsWeight, 
            wordFaceWeight,faceWordWeight,
            timeShortBeforeWeight, timeShortAfterWeight, timeShortDistance, 
            timeLongBeforeWeight, timeLongAfterWeight, timeLongDistance, 
            posInSentence, posInSentenceWidth, posInSentenceWeight, 
             preferredDuration, preferredDurationWidth, durationWeight):

    debug = False
    # get the lists we need
    if debug: print '***get lists for predicate', predicate
    _neighbors = normalizeItemScore(copy.copy(neighbors[predicate]))
    #print '\n_neighbors', _neighbors
    _wordsInSentence = normalizeItemScore(copy.copy(wordsInSentence[predicate]))
    #print '\n_wordsInSentence', _wordsInSentence
    _wordFace = normalizeItemScore(copy.copy(wordFace[predicate]))
    if debug: print '\n* faces that have said this word', _wordFace
    #print '\n_wordFace', _wordFace
    # temporary solution to using faces
    faces = [item[0] for item in _wordFace]
    if -1 in faces: faces.remove(-1)
    if faces == []: face = 0
    else: face = random.choice(faces)
    #print 'using face', face, 'we might want to update face/word selection', # USE WORDFACEWEIGHT TO SELECT FACE
    #print 'faceWord', faceWord
    try:_faceWord = normalizeItemScore(copy.copy(faceWord[face]))
    except:_faceWord = [] # if for some reason we can't find any faces
    if debug: print '\n_faceWord', _faceWord
    _similarWords = copy.copy(similarWords[predicate])
    _similarWords = list(np.add(scale(_similarWords, -1), 1.0)) # invert scores (keeping in 0 to 1 range)
    _similarWords = normalizeItemScore(formatAsMembership(_similarWords))
    _similarWords = zeroMe(predicate, _similarWords)
    if debug: print '\n_similarWords', _similarWords
    #print '\*similarWords*'
    #for k,v in similarWords.iteritems():
    #    print 'audio id %i has %i similar sounds'%(k, len(v))
    #    print v
    timeShortContextBefore, timeShortContextAfter = getTimeContext(predicate, timeShortDistance) 
    timeShortContextBefore = normalizeItemScore(timeShortContextBefore)
    timeShortContextAfter = normalizeItemScore(timeShortContextAfter)
    #print '\ntimeShortContextBefore',timeShortContextBefore
    #print '\ntimeShortContextAfter',timeShortContextAfter
    timeLongContextBefore, timeLongContextAfter = getTimeContext(predicate, timeLongDistance) 
    timeLongContextBefore = normalizeItemScore(timeLongContextBefore)
    timeLongContextAfter = normalizeItemScore(timeLongContextAfter)
    #print '\ntimeLongContextBefore',timeLongContextBefore
    #print '\ntimeLongContextAfter',timeLongContextAfter
    posInSentenceContext = getCandidatesFromContext(sentencePosition_item, posInSentence, posInSentenceWidth)
    #print 'posInSentence', posInSentence
    #print '\nposInSentenceContext', posInSentenceContext
    durationContext = getCandidatesFromContext(duration_word, preferredDuration, preferredDurationWidth)
    #print '\ndurationContext', durationContext
        
    #print 'generate lengths:', len(timeContextBefore), len(timeContextAfter), len(durationContext)
    # merge them
    if method == 'add': method = weightedSum
    if method == 'boundedAdd': method = boundedSum
    temp = method(_neighbors, neighborsWeight, _wordsInSentence, wordsInSentenceWeight)
    #temp = method(temp, 1.0, _wordFace, wordFaceWeight)
    #print 'temp1', temp
    temp = method(temp, 1.0, _faceWord, faceWordWeight)
    temp = weightedSum(temp, 1.0, _similarWords, similarWordsWeight)
    temp = weightedSum(temp, 1.0, timeShortContextBefore, timeShortBeforeWeight)
    temp = weightedSum(temp, 1.0, timeShortContextAfter, timeShortAfterWeight)
    temp = boundedSum(temp, 1.0, timeLongContextBefore, timeLongBeforeWeight)
    temp = boundedSum(temp, 1.0, timeLongContextAfter, timeLongAfterWeight)
    #print 'temp2', temp
    temp = weightedSum(temp, 1.0, posInSentenceContext, posInSentenceWeight)
    temp = method(temp, 1.0, durationContext, durationWeight) 
    #print 'generate temp', temp       
    # select the one with the highest score
    if len(temp) < 1:
        nextWord = predicate
        if debug: print '** WARNING: ASSOCIATION GENERATE HAS NO VALID ASSOCIATIONS, RETURNING PREDICATE'
    else:
        nextWord = select(temp, 'from_highest') # from_highest method introduce random selection between highest scores, BEWARE for confusion in debugging of scores/weights

    return nextWord


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

def getSimilarWords(predicate, distance):
    _similarWords = copy.copy(similarWords[predicate])
    print 'getSimilarWords', predicate, distance, _similarWords 
    try:
        _similarWords = formatAsMembership(_similarWords)
        _similarWords = zeroMe(predicate, _similarWords)
        simIds = []
        for item in _similarWords:
            if item[1] < distance: simIds.append(item)
        for item in simIds: item.reverse()
        simIds.sort() # get the best rhymes
        simIds = [ item[1] for item in simIds ]
    except Exception, e:
        print e, 'getSimilarWords failed'
        simIds = [0]
    return simIds

def getFaceResponse(face):
    print 'getFaceResponse', face, faceWord
    words = [item[0] for item in copy.copy(faceWord[face])]
    other_faces = faceWord.keys()
    other_faces.remove(face)
    for f in other_faces:
        for item in faceWord[f]:
            if len(words)>1:
                if item[0] in words: words.remove(item[0])
    print 'words', words
    return random.choice(words)

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
    a = list(np.array(a)*scale)
    return a

def clip(a, clipVal):
    if clipVal >= 0:
        a = list(np.clip(a, 0, clipVal))
    else:
        a = list(np.array(np.clip(a, 0, abs(clipVal)))*-1)
    return a

def formatAsMembership(a):
    i = 0
    membership = []
    for item in a:
        membership.append([i,item])
        i += 1
    return membership
    
def zeroMe(me, a):
    for item in a:
        if item[0] == me: item[1] = 0.0
    return a

def select(items, method):
    words = [items[i][0] for i in range(len(items))]
    scores = [items[i][1] for i in range(len(items))]
    if method == 'highest':
        return words[scores.index(max(scores))]
    if method == 'from_highest': # if several items has the same (highest) score, shuffle them randomly
        highscore = max(scores)
        temp = copy.copy(scores)
        temp.sort()
        while highscore in temp: temp.remove(highscore)
        try: next_highest = max(temp)
        except: next_highest = highscore # if we did not have duplicates, just quietly select the one highest
        high_range = highscore - next_highest
        for i in range(len(scores)):
            if scores[i] == highscore: scores[i] += random.random()*high_range*0.99
        new_high = max(scores)
        return words[scores.index(max(scores))]
    elif method == 'lowest':
        return words[scores.index(min(scores))]
    else:
        random.choice(words)

def pushCurrentSettings():
    global numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight
    global timeShortBeforeWeight,timeShortAfterWeight,timeShortDistance,timeLongBeforeWeight,timeLongAfterWeight,timeLongDistance,durationWeight,posInSentenceWeight
    global neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2
    global timeShortBeforeWeight2,timeShortAfterWeight2,timeShortDistance2,timeLongBeforeWeight2,timeLongAfterWeight2,timeLongDistance2,durationWeight2,posInSentenceWeight2   
    currentSettings.append([numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight,\
    timeShortBeforeWeight,timeShortAfterWeight,timeShortDistance,timeLongBeforeWeight,timeLongAfterWeight,timeLongDistance,durationWeight,posInSentenceWeight,\
    neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2,\
    timeShortBeforeWeight2,timeShortAfterWeight2,timeShortDistance2,timeLongBeforeWeight2,timeLongAfterWeight2,timeLongDistance2,durationWeight2,posInSentenceWeight2])

def popCurrentSettings():
    global numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight
    global timeShortBeforeWeight,timeShortAfterWeight,timeShortDistance,timeLongBeforeWeight,timeLongAfterWeight,timeLongDistance,durationWeight,posInSentenceWeight
    global neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2
    global timeShortBeforeWeight2,timeShortAfterWeight2,timeShortDistance2,timeLongBeforeWeight2,timeLongAfterWeight2,timeLongDistance2,durationWeight2,posInSentenceWeight2   
    numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight,\
    timeShortBeforeWeight,timeShortAfterWeight,timeShortDistance,timeLongBeforeWeight,timeLongAfterWeight,timeLongDistance,durationWeight,posInSentenceWeight,\
    neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2,\
    timeShortBeforeWeight2,timeShortAfterWeight2,timeShortDistance2,timeLongBeforeWeight2,timeLongAfterWeight2,timeLongDistance2,durationWeight2,posInSentenceWeight2 = currentSettings.pop(0)

def setParam(param,value):
    #param is a string, so we must compile the statement to set the variable
    print 'setParam:', param, value
    global numWords,neighborsWeight,wordsInSentenceWeight,similarWordsWeight,wordFaceWeight,faceWordWeight
    global timeShortBeforeWeight,timeShortAfterWeight,timeShortDistance,timeLongBeforeWeight,timeLongAfterWeight,timeLongDistance,durationWeight,posInSentenceWeight
    global neighborsWeight2,wordsInSentenceWeight2,similarWordsWeight2,wordFaceWeight2,faceWordWeight2
    global timeShortBeforeWeight2,timeShortAfterWeight2,timeShortDistance2,timeLongBeforeWeight2,timeLongAfterWeight2,timeLongDistance2,durationWeight2,posInSentenceWeight2   

    if param == 'numWords': numWords = int(value)
    if param == 'neighborsWeight':neighborsWeight = float(value)
    if param == 'wordsInSentenceWeight':wordsInSentenceWeight = float(value)
    if param == 'similarWordsWeight':similarWordsWeight = float(value)
    if param == 'wordFaceWeight':wordFaceWeight = float(value)
    if param == 'faceWordWeight':faceWordWeight = float(value)
    if param == 'timeShortBeforeWeight':timeShortBeforeWeight = float(value)
    if param == 'timeShortAfterWeight':timeShortAfterWeight = float(value)
    if param == 'timeShortDistance':timeShortDistance = float(value)
    if param == 'timeLongBeforeWeight':timeLongBeforeWeight = float(value)
    if param == 'timeLongAfterWeight':timeLongAfterWeight = float(value)
    if param == 'timeLongDistance':timeLongDistance = float(value)
    if param == 'durationWeight':durationWeight = float(value)
    if param == 'posInSentenceWeight':posInSentenceWeight = float(value)
    if param == 'neighborsWeight2':neighborsWeight2 = float(value)
    if param == 'wordsInSentenceWeight2':wordsInSentenceWeight2 = float(value)
    if param == 'similarWordsWeight2':similarWordsWeight2 = float(value)
    if param == 'wordFaceWeight2':wordFaceWeight2 = float(value)
    if param == 'faceWordWeight2':faceWordWeight2 = float(value)
    if param == 'timeShortBeforeWeight2':timeShortBeforeWeight2 = float(value)
    if param == 'timeShortAfterWeight2':timeShortAfterWeight2 = float(value)
    if param == 'timeShortDistance2':timeShortDistance2 = float(value)
    if param == 'timeLongBeforeWeight2':timeLongBeforeWeight2 = float(value)
    if param == 'timeLongAfterWeight2':timeLongAfterWeight2 = float(value)
    if param == 'timeLongDistance2':timeLongDistance2 = float(value)
    if param == 'durationWeight2':durationWeight2 = float(value)
    if param == 'posInSentenceWeight2':posInSentenceWeight2 = float(value)
    if param == 'zero':
        numWords = 5
        neighborsWeight = 0.0
        wordsInSentenceWeight = 0.0
        similarWordsWeight = 0.0
        wordFaceWeight = 0.0
        faceWordWeight = 0.1
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.0
        timeShortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.0
        timeLongDistance = 120.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.0
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = 0.0
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.1
        timeShortBeforeWeight2 = 0.0
        timeShortAfterWeight2 = 0.0
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.0
        timeLongAfterWeight2 = 0.0
        timeLongDistance2 = 120.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0

    if param == 'identical':
        numWords = 5
        neighborsWeight = 0.04
        wordsInSentenceWeight = 0.0
        similarWordsWeight = 0.3
        wordFaceWeight = 0.0
        faceWordWeight = 0.0
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.7
        timeShortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.1
        timeLongDistance = 120.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.04
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = 0.3
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.0
        timeShortBeforeWeight2 = 0.0
        timeShortAfterWeight2 = 0.7
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.0
        timeLongAfterWeight2 = 0.1
        timeLongDistance2 = 120.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0

    if param == 'nidentical':
        numWords = 5
        neighborsWeight = 0.04
        wordsInSentenceWeight = 0.0
        similarWordsWeight = 0.3
        wordFaceWeight = 0.0
        faceWordWeight = 0.1
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.7
        timeShortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.1
        timeLongDistance = 120.0
        durationWeight = 0.3
        posInSentenceWeight = 0.3
        neighborsWeight2 = 0.04
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = 0.3
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.1
        timeShortBeforeWeight2 = 0.0
        timeShortAfterWeight2 = 0.7
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.0
        timeLongAfterWeight2 = 0.1
        timeLongDistance2 = 120.0
        durationWeight2 = 0.3
        posInSentenceWeight2 = 0.3

    if param == 'backwards':
        numWords = 5
        neighborsWeight = 0.2
        wordsInSentenceWeight = 0.1
        similarWordsWeight = 0.1
        wordFaceWeight = 0.0
        faceWordWeight = 0.0
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.8
        timeShortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.15
        timeLongDistance = 120.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.2
        wordsInSentenceWeight2 = 0.1
        similarWordsWeight2 = 0.1
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.0
        timeShortBeforeWeight2 = 0.8
        timeShortAfterWeight2 = 0.0
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.15
        timeLongAfterWeight2 = 0.0
        timeLongDistance2 = 120.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0
    
    if param == 'similar':
        numWords = 5
        neighborsWeight = 0.2
        wordsInSentenceWeight = 0.1
        similarWordsWeight = 0.1
        wordFaceWeight = 0.0
        faceWordWeight = 0.1
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.8
        timeshortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.1
        timeLongDistance = 120.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.0
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = 0.9
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.1
        timeShortBeforeWeight2 = 0.0
        timeShortAfterWeight2 = 0.0
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.0
        timeLongAfterWeight2 = 0.0
        timeLongDistance2 = 120.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0
    
    if param == 'notsimilar':
        numWords = 5
        neighborsWeight = 0.2
        wordsInSentenceWeight = 0.1
        similarWordsWeight = 0.1
        wordFaceWeight = 0.0
        faceWordWeight = 0.1
        timeShortBeforeWeight = 0.0
        timeShortAfterWeight = 0.8
        timeShortDistance = 15.0
        timeLongBeforeWeight = 0.0
        timeLongAfterWeight = 0.1
        timeLongDistance = 120.0
        durationWeight = 0.0
        posInSentenceWeight = 0.0
        neighborsWeight2 = 0.0
        wordsInSentenceWeight2 = 0.0
        similarWordsWeight2 = -0.9
        wordFaceWeight2 = 0.0
        faceWordWeight2 = 0.1
        timeShortBeforeWeight2 = 0.0
        timeShortAfterWeight2 = 0.0
        timeShortDistance2 = 15.0
        timeLongBeforeWeight2 = 0.0
        timeLongAfterWeight2 = 0.0
        timeLongDistance2 = 120.0
        durationWeight2 = 0.0
        posInSentenceWeight2 = 0.0


if __name__ == '__main__':
    print 'run as main'
