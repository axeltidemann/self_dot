#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with markov models
'''

import random
import loadText

loadText.importFromFile('snowflakes_db.txt')
words = loadText.words
neighbors = loadText.neighbors


## testing
if __name__ == '__main__':
    #print '\n\n***\n'
    predicate = random.choice(words.keys())
    sentence = [predicate]
    print 'predicate:', predicate
    candidates = words[sentence[-1]]
    nextTo = neighbors[sentence[-1]]
    i_c_n = set(candidates).intersection(set(nextTo))
    if i_c_n == set([]): 
        i_c_n = candidates
        print 'no intersection of candidates and neighbors (1)'
    association = random.choice(tuple(i_c_n))
    print 'association', association
    sentence.append(association)
    print 'the current sentence is:', sentence
    predicate = association
    for i in range(4):
        c1 = set(words[sentence[-1]])
        c2 = set(words[sentence[-2]])
        nextTo = neighbors[sentence[-1]]
        i_1_2 = c1.intersection(c2) # so, not 2nd order Markov, but intersection of two sets of candidates
        i_1_2_n = i_1_2.intersection(nextTo) # 
        if i_1_2 == set([]): 
            print 'no intersection of both candidates'
            i_c1_n = c1.intersection(nextTo)
            i_c2_n = c2.intersection(nextTo)
            if i_c1_n > i_c2_n:
                association = random.choice(tuple(i_c1_n))
            else:
                association = random.choice(tuple(i_c2_n))
        elif i_1_2_n == set([]): 
            print 'no intersection of both candidates and neighbors'
        else:
            association = random.choice(tuple(i_1_2_n))
        print '\t association:', association
        sentence.append(association)
        print 'the current sentence is:', sentence
        