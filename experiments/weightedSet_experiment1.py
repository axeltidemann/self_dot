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
    scoreA = normScale([a[i][1] for i in range(len(a))],weightA)
    scoreB = normScale([b[i][1] for i in range(len(b))],weightB)
    removeFromB = []
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        if itemA in itemsB:
            j = itemsB.index(itemA)
            c.append([itemsA[i], scoreB[j] + scoreA[i]])    # add scores and put item in c
            removeFromB.append(j)                       # removing used elements...
        else:
            c.append([itemsA[i], scoreA[i]])         
    removeFromB.sort()
    removeFromB.reverse()
    for i in removeFromB:
        b.remove(b[i])
    if len(b) > 0:
            c.append([b[i][0], b[i][1]*weightB])
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
    scoreA = normScale([a[i][1] for i in range(len(a))],weightA)
    scoreB = normScale([b[i][1] for i in range(len(b))],weightB)
    removeFromB = []
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        if itemA in itemsB:
            j = itemsB.index(itemA)
            c.append([itemsA[i], scoreB[j] * scoreA[i]])    # multiply scores and put item in c
            removeFromB.append(j)                       # removing used elements...
        else:
            c.append([itemsA[i], defaultScale(scoreA[i])])   
    removeFromB.sort()
    removeFromB.reverse()
    for i in removeFromB:
        b.remove(b[i])
    if len(b) > 0:
        c.append([b[i][0], defaultScale(b[i][1]*weightB)])
    return c

def normalize(a):
    return a
    if a != []:
        highest = float(max(a))
        if highest != 0:
            for i in range(len(a)):
                a[i] /= highest
    return a
     
def normScale(a, scale):
    a = normalize(a)
    a = list(numpy.array(a)*scale)
    return a


def select(items, method):
    words = [items[i][0] for i in range(len(items))]
    scores = [items[i][1] for i in range(len(items))]
    if method == 'highest':
        return words[scores.index(max(scores))]
    elif method == 'lowest':
        return words[scores.index(min(scores))]
    else:
        random.choice(words)
        

def generate(predicate, method):
    # get the lists we need
    neighbors = l.neighbors[predicate]
    wordsInSentence = l.wordsInSentence[predicate]
    similarWords = l.similarWords[predicate]
    neighborsWeight = 0.000000000000000111022
    wordsInSentenceWeight = 0#1#1.0
    similarWordsWeight = 1#0.000000000000001
    #print 'lengths', len(neighbors), len(wordsInSentence), len(similarWords)
    if method == 'multiply':
        # multiply (soft intersection)
        temp = weightedMultiply(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        #print 'templength', len(temp)
        temp = weightedMultiply(temp, 1.0, similarWords, similarWordsWeight)
        #print 'templength', len(temp)
    if method == 'add':
        # add (union)
        temp = weightedSum(neighbors, neighborsWeight, wordsInSentence, wordsInSentenceWeight)
        #print 'templength', len(temp)
        temp = weightedSum(temp, 1.0, similarWords, similarWordsWeight)
        #print 'templength', len(temp)
    # select the one with the highest score
    nextWord = select(temp, 'highest')
    return nextWord
    
def testSentence(method):
    l.importFromFile('association_test_db_full.txt', 1)#minimal_db.txt')#roads_articulation_db.txt')#
    predicate = random.choice(list(l.words))
    print 'predicate', predicate
    sentence = [predicate]
    for i in range(12):
        predicate = generate(predicate, method)
        sentence.append(predicate)
        print 'sentence', sentence

def testMerge():
    a = [['world', 1.0],['you', 0.5], ['there', 0.3],['near', 0.5]] #near only exist in this set
    b = [['world', 0.3],['you', 1.0], ['there', 0.2],['here', 0.8],['nowhere',0.1]] #here and nowhere only exist in this set
    c = [['world', 1.0],['you', 0.5], ['were', 0.4]] #were only exist in this set, and here is not here
    print '\n** sum **'
    t = weightedSum(a,1,b,1)
    t = weightedSum(c,1,t,1)
    for item in t:
        print item
    print 'select:', select(t, 'highest')
    
    print '\n** multiply **'
    t = weightedMultiply(a,1,b,1)
    t = weightedMultiply(c,1,t,1)
    for item in t:
        print item
    print 'select:', select(t, 'highest')

## testing
if __name__ == '__main__':
    testSentence('add')
    #testMerge()
