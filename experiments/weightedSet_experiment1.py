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
    itemsBcopy = copy.copy(itemsB)
    scoreA = normalize([a[i][1] for i in range(len(a))])
    scoreB = normalize([b[i][1] for i in range(len(b))])
    for i in range(len(itemsA)):
        itemA = itemsA[i]
        if itemA in itemsB:
            j = itemsB.index(itemA)
            print itemsA[i], b[j], a[i] ### something can appparently go wrong here ###
            c.append([itemsA[i], b[j][1] * a[i][1]])    # multiply scores and put item in c
            remIndex = itemsBcopy.index(itemA)          # update B, by...
            b.remove(b[remIndex])                       # removing used elements...
            itemsBcopy.remove(itemsBcopy[remIndex])     # and update the copy to keep indices consistent
        else:  
            c.append([itemsA[i], replacementValue])     # replacementValue may get a smarter implementation
        for i in range(len(b)):
            c.append([b[i][0], replacementValue])
    return c
            
def normalize(a):
    highest = max(a)
    for i in range(len(a)):
        a[i] /= highest
    return a
        

def generate(predicate):
    # get the lists we need
    neighbors = l.neighbors[predicate]
    wordsInSentence = l.wordsInSentence[predicate]
    similarWords = l.similarWords[predicate]
    # intersection
    temp = weightedIntersection(neighbors, wordsInSentence, 0)
    # select the one with the highest score

## testing
if __name__ == '__main__':
    l.importFromFile('minimal_db.txt')
    predicate = random.choice(list(l.words))
    sentence = [predicate]
    '''
    for i in range(8):
        predicate = generate(predicate)
        sentence.append(predicate)
        print sentence
    '''
    print generate(predicate)
        