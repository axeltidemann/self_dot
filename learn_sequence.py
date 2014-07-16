#!/usr/bin/python
# -*- coding: latin-1 -*-

''' Learns a sequence of symbols. 

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL
'''

import Oger
import mdp
import numpy as np

def _index_to_text(idxs, table):
    return [ table[np.argmax(row)] for row in idxs ]

def learn(sequence, freerun_steps=None, repeat=10):
    unique = list(set(sequence))
    train_signals = np.zeros((len(sequence), len(unique)))
    for row, word in zip(train_signals, sequence):
        row[unique.index(word)] = 1

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, 
                                              leak_rate=0.4, 
                                              input_scaling=.05, 
                                              bias_scaling=.2, 
                                              reset_states=False)
    readout = Oger.nodes.RidgeRegressionNode()

    if freerun_steps is None:
        freerun_steps = len(sequence)

    train_signals = np.tile(train_signals, [repeat,1])

    flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=freerun_steps)

    flow.train([[], [[ train_signals ]]])

    return flow, train_signals, unique

    # Trying with random end word
    test = train_signals[np.random.choice(len(sequence))]
    print 'Start word: {}'.format(_index_to_text(np.atleast_2d(test), unique))
    test_signals = np.tile(test, [freerun_steps+1,1])
    return _index_to_text(flow(test_signals)[-freerun_steps:], unique)
    
    return _index_to_text(flow(train_signals)[-freerun_steps:], unique)
    
if __name__ == '__main__':
    txt = open('sequence.txt')
    symbols = []
    for line in txt.readlines():
        symbols.extend(line.split())
    net, train_signals, dictionary = learn(symbols)


    dummy = np.empty( (len(symbols), train_signals.shape[1]) )
    dummy[:] = np.nan
    print 'Regenerating entire learned sequence, feeding the entire sequence as input. This should yield stable behaviour.'
    print _index_to_text(net(
        np.vstack((train_signals, dummy)))[-len(symbols):],
        dictionary)

    print 'Predicting rest of learned sequence. The network is fed up until this word, i.e. this is the input:'
    marker = np.random.choice(len(symbols))
    up_until = train_signals[:marker]
    print _index_to_text(up_until, dictionary)
    print _index_to_text(net(
        np.vstack(( up_until, dummy )))[marker:len(symbols)],
        dictionary)
    
    random_word = np.atleast_2d(train_signals[np.random.choice(len(symbols))])
    print 'Starting with a randomly selected word, feeding just this one word to the network: {}\
    This should yield unstable sequences.'.format(_index_to_text(random_word, dictionary))
    print _index_to_text(net(
        np.vstack((random_word, dummy)))[1:],
        dictionary)
