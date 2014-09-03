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
import loadDbWeightedSet as l


def weightedIntersection(a_, b_, replacementValue):
    c = []
    if len(a_) > len(b_):
        a = a_
        b = b_
    else:
        a = b_
        b = a_
    itemsA = [a[i][0] for i in range(len(a))]
    itemsB = [b[i][0] for i in range(len(b))]
    scoreA = normalize([a[i][1] for i in range(len(a))])
    scoreB = normalize([b[i][1] for i in range(len(b))])
    removeFromB = []
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        if itemA in itemsB:
            j = itemsB.index(itemA)
            c.append([itemsA[i], b[j][1] * a[i][1]])    # multiply scores and put item in c
            removeFromB.append(j)                       # removing used elements...
        else:  
            c.append([itemsA[i], replacementValue])     # replacementValue may get a smarter implementation
    removeFromB.sort()
    removeFromB.reverse()
    for i in removeFromB:
        b.remove(b[i])
    for i in range(len(b)):
        c.append([b[i][0], replacementValue])
    return c
            
def normalize(a):
    if a != []:
        highest = max(a)
        if highest != 0:
            for i in range(len(a)):
                a[i] /= highest
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
        

def generate(predicate):
    # get the lists we need
    neighbors = l.neighbors[predicate]
    wordsInSentence = l.wordsInSentence[predicate]
    similarWords = l.similarWords[predicate]
    # intersection
    temp = weightedIntersection(neighbors, wordsInSentence, 0)
    temp = weightedIntersection(temp, similarWords, 0)
    # select the one with the highest score
    nextWord = select(temp, 'highest')
    return nextWord
    
## testing
if __name__ == '__main__':
    l.importFromFile('association_test_db_full.txt')#minimal_db.txt')#roads_articulation_db.txt')#
    predicate = random.choice(list(l.words))
    print 'predicate', predicate
    sentence = [predicate]
    for i in range(12):
        predicate = generate(predicate)
        sentence.append(predicate)
        print 'sentence', sentence
    '''
    predicate = generate(predicate)
    print 'generator', predicate
    '''        