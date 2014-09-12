#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with weighted sets.

Somewhat like fuzzy sets in that items can have partial membership in a set.
e.g. for similarity measure and for context based on distance in time.
The general format for a set here is [[item, score],[item, score],...]
This is wrapped in a dictionary for each dimension (e.g. similarity),
where the format is {word:[[item, score],[item, score],...], ...},
allowing "query the dict for a weighted set of alternatives for items to associate with word"

@author: Oeyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL

'''

import random
import copy
import numpy
import math
import loadDbWeightedSet as l


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

def defaultScale(x):
    '''
    For use in weightedMultiply.
    For now, we use an empirical adjustment trying to achive values in a range that could be produced by a*b
    and also stay within approximately the same average range as random*random*0.5
    '''
    #return (math.sqrt(x*0.1)+math.pow(x*0.5, 2))*0.5
    return x*0.25

def weightedMultiply(a_, weightA_, b_, weightB_):
    '''
    Multiply the score of items found in both sets.
    If an item is found only in one of the sets, we need to find a way to scale the value appropriately,
    see defaultScale for more info
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
    #print 'weightedMultiply', min(scoreA), max(scoreA), min(scoreB), max(scoreB)
    removeFromB = set()
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        a_in_b = 0
        itemsBB = copy.copy(itemsB)
        scoreBB = copy.copy(scoreB)
        while itemA in itemsBB:
            a_in_b = 1
            j = itemsBB.index(itemA)
            c.append([itemsA[i], scoreBB[j] * scoreA[i]])    # add scores and put item in c
            itemsBB[j] = '_'      # hide used elements but keep indexing unmodified
            scoreBB[j] = '_'
            removeFromB.add(j)   # mark this one to be removed later
        if not a_in_b:
            c.append([itemsA[i], defaultScale(scoreA[i])])         
    removeFromB = list(removeFromB)
    removeFromB.sort()  
    removeFromB.reverse() # ... so we can use pop(n), starting from the end
    for i in removeFromB:
        gone = b.pop(i)
    for i in range(len(b)):
        c.append([b[i][0], defaultScale(b[i][1]*weightB)])
    c = normalizeItemScore(c) # to avoid accumulatively less weight when merging more tthan two sets
    return c

def weightedMultiplySqrt(a_, weightA_, b_, weightB_):
    '''
    Multiply the score of items found in both sets.
    To compensate for items found only in ome of the sets, 
    we take the square of the multiplication for items found in both sets,
    and divide by 2 for items found in only one of the sets. 
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
    #print 'wMultiplySqrt', min(scoreA), max(scoreA), min(scoreB), max(scoreB)
    removeFromB = set()
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        a_in_b = 0
        itemsBB = copy.copy(itemsB)
        scoreBB = copy.copy(scoreB)
        while itemA in itemsBB:
            a_in_b = 1
            j = itemsBB.index(itemA)
            c.append([itemsA[i], math.sqrt(scoreBB[j] * scoreA[i])])    # add scores and put item in c
            itemsBB[j] = '_'      # hide used elements but keep indexing unmodified
            scoreBB[j] = '_'
            removeFromB.add(j)   # mark this one to be removed later
        if not a_in_b:
            c.append([itemsA[i], scoreA[i]*0.5])         
    removeFromB = list(removeFromB)
    removeFromB.sort()  
    removeFromB.reverse() # ... so we can use pop(n), starting from the end
    for i in removeFromB:
        gone = b.pop(i)
    for i in range(len(b)):
        c.append([b[i][0], b[i][1]*0.5])
    c = normalizeItemScore(c) # to avoid accumulatively less weight when merging more tthan two sets
    return c

def normalize(a):
    if a != []:
        highest = float(max(a))
        if highest != 0:
            for i in range(len(a)):
                a[i] /= highest
    return a
     
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
    a = list(numpy.clip(a, 0, clipVal))
    return a


def select(items, method):
    words = [items[i][0] for i in range(len(items))]
    scores = [items[i][1] for i in range(len(items))]
    #print 'select', items
    #print 'select item', scores.index(max(scores))
    #print 'select max:', max(scores), min(scores)
    if method == 'highest':
        return words[scores.index(max(scores))]
    elif method == 'lowest':
        return words[scores.index(min(scores))]
    else:
        random.choice(words)

def getTimeContext(predicate, distance):
    '''
    * Look up the time(s) when predicate has been used (l.wordTime),
    * Look up these times (l.time_word), generating a list of words
    appearing within a desired distance in time from each of these time points.
    * Make a new list of [word, distance], with the distance inverted (use maxDistance - distance) and normalized.
    Invert items [distance,word] and sort, invert items again.
    This list will have items (words) sorted from close in time to far in time, retaining a normalized score 
    for how far in time from the predicate each word has occured.
    '''
    timeWhenUsed = l.wordTime[predicate]
    quantize = 0.01
    iquantize = 1/quantize
    timeContextBefore = []
    timeContextAfter = []
    for t in timeWhenUsed:
        startIndex = -1
        endIndex = -1
        for i in range(len(l.time_word)):
            if (l.time_word[i][0] > (t-distance)) and (startIndex == -1):
                startIndex = i
            if (l.time_word[i][0] > (t+distance)):
                endIndex = i
                break
        if startIndex == -1: startIndex = 0
        if endIndex == -1: endIndex = len(l.time_word)
        for j in range(startIndex, endIndex):
            if l.time_word[j][0]-t > 0 : # do not include the query word
                timeContextAfter.append(((int(l.time_word[j][0]*iquantize)-int(t*iquantize)),l.time_word[j][1]))
            if l.time_word[j][0]-t < 0 : # do not include the query word
                timeContextBefore.append((int(t*iquantize)-(int(l.time_word[j][0]*iquantize)),l.time_word[j][1]))
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
    

def testgenerate(predicate, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance):
    # get the lists we need
    neighbors = l.neighbors[predicate]
    wordsInSentence = l.wordsInSentence[predicate]
    similarWords = normalize(copy.copy(l.similarWords[predicate]))
    timeContextBefore, timeContextAfter = getTimeContext(predicate, timeDistance) 
    temp = weightedSum(neighbors, neighborsWeight, timeContextBefore, timeBeforeWeight)
    return temp


def generate(predicate, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance):
    # get the lists we need
    neighbors = normalizeItemScore(copy.copy(l.neighbors[predicate]))
    wordsInSentence = normalizeItemScore(copy.copy(l.wordsInSentence[predicate]))
    similarWords = normalizeItemScore(copy.copy(l.similarWords[predicate]))
    timeContextBefore, timeContextAfter = getTimeContext(predicate, timeDistance) 
    timeContextBefore = normalizeItemScore(timeContextBefore)
    timeContextAfter = normalizeItemScore(timeContextAfter)
    if method == 'multiply':
        temp = weightedMultiply(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        temp = weightedMultiply(temp, 1.0, similarWords, similarWordsWeight)
        if len(timeContextBefore) > 0:
            temp = weightedMultiply(temp, 1.0, timeContextBefore, timeBeforeWeight)
        if len(timeContextAfter) > 0:
            temp = weightedMultiply(temp, 1.0, timeContextAfter, timeAfterWeight)
    if method == 'multiplySqrt':
        temp = weightedMultiplySqrt(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        temp = weightedMultiplySqrt(temp, 1.0, similarWords, similarWordsWeight)
        if len(timeContextBefore) > 0:
            temp = weightedMultiplySqrt(temp, 1.0, timeContextBefore, timeBeforeWeight)
        if len(timeContextAfter) > 0:
            temp = weightedMultiplySqrt(temp, 1.0, timeContextAfter, timeAfterWeight)
    if method == 'add':
        temp = weightedSum(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        temp = weightedSum(temp, 1.0, similarWords, similarWordsWeight)
        temp = weightedSum(temp, 1.0, timeContextBefore, timeBeforeWeight)
        temp = weightedSum(temp, 1.0, timeContextAfter, timeAfterWeight)
    if method == 'boundedAdd':
        temp = boundedSum(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        temp = boundedSum(temp, 1.0, similarWords, similarWordsWeight)
        if len(timeContextBefore) > 0:
            temp = boundedSum(temp, 1.0, timeContextBefore, timeBeforeWeight)
        if len(timeContextAfter) > 0:
            temp = boundedSum(temp, 1.0, timeContextAfter, timeAfterWeight)
    # select the one with the highest score
    nextWord = select(temp, 'highest')
    return nextWord
    
def testSentence(method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance):
    l.importFromFile('association_test_db_short.txt', 1)#association_test_db_full.txt', 1)#minimal_db.txt')#roads_articulation_db.txt')#
    predicate = 'to'#'parents'#random.choice(list(l.words))
    print 'predicate', predicate
    sentence = [predicate]
    for i in range(8):
        predicate = generate(predicate, method, neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance)
        sentence.append(predicate)
    print 'sentence', sentence

def testMerge():
    a = [['world', 1.0],['you', 0.5], ['there', 0.3],['near', 0.5]] #near only exist in this set
    b = [['world', 0.3],['you', 1.0], ['there', 0.2],['here', 0.8],['nowhere',0.1]] #here and nowhere only exist in this set
    c = [['world', 1.0],['you', 0.5], ['were', 0.4]] #were only exist in this set, and here is not here
    d = [['world', 1.0],['you', 0.5], ['were', 0.4], ['world', 0.1]] #as c but with a duplicate 'world'
    methods = [weightedSum, boundedSum, weightedMultiply, weightedMultiplySqrt]
    labels = ['weightedSum', 'boundedSum', 'weightedMultiply', 'weightedMultiplySqrt']
    weightA = 0.5
    weightB = 0.5
    weightC = 0.5
    weightD = 0.5
    for i in range(len(methods)):
        print '\n** %s **'%labels[i]
        t = methods[i](a,weightA,b,weightB)
        print 'a+b', t
        t = methods[i](d,weightD,t,1)
        print 'd+t...'
        for item in t:
            print item
        t = methods[i](c,weightC,t,1)
        print 'c+t...'
        for item in t:
            print item

        print 'select:', select(t, 'highest')
    
## testing
if __name__ == '__main__':
    '''
    l.importFromFile('association_test_db_full.txt', 1)#minimal_db.txt')#roads_articulation_db.txt')#
    predicate = 'to'#'parents'#random.choice(list(l.words))
    t = testgenerate(predicate, 'add', 0.0, 0.1, 0.0, 0.1, 0.1, 5.0) 
    print t
    '''
    ## TEST ADD
    #neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance
    # OK example
    #testSentence('add', 0.0, 0.0, 0.2, 0.0, 1.0, 5.0) 
    # careful neighborsWeight must be quite low, or it will create repetitive output 
    #testSentence('add', 0.1, 0.0, 0.2, 0.0, 1.0, 5.0) 
    # the same, even more so, applies to wordsInSentenceWeight
    #testSentence('add', 0.0, 0.1, 0.2, 0.0, 1.0, 5.0) 
    # the same also applies to timeBeforeWeight
    #testSentence('add', 0.0, 0.0, 0.2, 0.1, 1.0, 5.0) 
    
    ## TEST BOUNDED ADD
    #neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance
    #testSentence('boundedAdd', 0.0, 0.0, 0.1, 0.1, 0.0, 5.0) # moves backwards in time
    #testSentence('boundedAdd', 0.1, 0.1, 0.1, 0.1, 0.9, 5.0) 
    #testSentence('boundedAdd', 0.4, 0.1, 0.1, 0.1, 0.9, 5.0) 
    #testSentence('boundedAdd', 0.1, 0.4, 0.1, 0.1, 0.9, 5.0) 
    
    ## TEST MULTIPLY
    #neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance
    #testSentence('multiply', 0.7, 0.2, 0.1, 0.0, 0.5, 5.0) 
    # overall, multiply is more sensitive to specific weighting, more difficult, less intuitive weights
    
    ## TEST MULTIPLY SQRT
    #neighborsWeight, wordsInSentenceWeight, similarWordsWeight, timeBeforeWeight, timeAfterWeight, timeDistance
    #testSentence('multiplySqrt', 0.1, 0.1, 0.7, 0.1, 0.5, 5.0) 
    # overall, multiply is more sensitive to specific weighting, more difficult, less intuitive weights
    
    #testMerge()
