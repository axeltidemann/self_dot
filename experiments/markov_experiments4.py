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
    for i in range(8):
        c1 = set(words[sentence[-1]])
        nextTo = neighbors[sentence[-1]]
        i_c_n = c1.intersection(nextTo) # 
        if i_c_n == set([]): 
            print 'no intersection of candidates and neighbors'
            association = random.choice(nextTo)
        else:
            association = random.choice(tuple(i_c_n))
        print '\t association:', association
        sentence.append(association)
        print 'the current sentence is:', sentence
        