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
    for i in range(4):
        candidates = words[predicate]
        association = random.choice(candidates)
        print '\t association:', association
        sentence.append(association)
        print 'the current sentence is:', sentence
        predicate = association
