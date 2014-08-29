#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with markov models
'''

import random
import loadText

loadText.importFromFile('snowflakes_db.txt')
words = loadText.words


## testing
if __name__ == '__main__':
    #print '\n\n***\n'
    predicate = random.choice(words.keys())
    sentence = [predicate]
    print 'predicate:', predicate
    candidates = words[sentence[-1]]
    association = random.choice(candidates)
    sentence.append(association)
    print 'the current sentence is:', sentence
    predicate = association
    for i in range(4):
        c1 = set(words[sentence[-1]])
        c2 = set(words[sentence[-2]])
        i_1_2 = c1.intersection(c2) # so, not 2nd order Markov, but intersection of two sets of candidates
        association = random.choice(tuple(i_1_2))
        print '\t association:', association
        sentence.append(association)
        print 'the current sentence is:', sentence
        predicate = association
        