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

def index_to_text(idxs, table):
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

    
if __name__ == '__main__':
    txt = open('sequence.txt')
    symbols = []
    for line in txt.readlines():
        symbols.extend(line.split())
    net, train_signals, dictionary = learn(symbols)

    dummy = np.empty( (len(symbols), train_signals.shape[1]) )
    dummy[:] = np.nan # NaNs spread like the plauge, we therefore use them to check for programming errors.

    print 'Regenerating entire learned sequence, feeding the entire sequence as input. This should yield stable behaviour.'
    dummy[0] = train_signals[0] # Due to a bug (?) in Oger, the first non-teacher forced signal must be manually set.
    print index_to_text(net(
        np.vstack((train_signals, dummy)))[-len(symbols):],
        dictionary)

    print '\nPredicting rest of learned sequence. The network is fed up until this word, i.e. this is the input:'
    marker = np.random.choice(len(symbols))
    up_until = train_signals[:marker]
    print index_to_text(up_until, dictionary)
    print 'This is the output:'
    dummy[0] = train_signals[marker]
    print index_to_text(net(
        np.vstack(( up_until, dummy )))[marker:len(symbols)],
        dictionary)
    
    random_word = np.random.choice(len(symbols))
    dummy[0] = train_signals[random_word]
    print '\nStarting with a randomly selected word, feeding just this one word to the network: {}'.format(index_to_text(np.atleast_2d(train_signals[random_word]), dictionary))
    print 'This should yield unstable sequences:'
    print index_to_text(net(
        np.vstack((np.atleast_2d(train_signals[random_word]), dummy)))[1:],
        dictionary)
    
    print '\nExample of secondary guesses of symbols. Tested to output entire sequence, but instead secondary guesses are displayed.'
    dummy[0] = train_signals[0] 
    output = net(np.vstack(( train_signals, dummy )))[-len(symbols):]
    
    for row in output:
        row[np.argmax(row)] = np.min(row)

    print index_to_text(output, dictionary)

    print '\nExample of negated guesses of symbols. Tested to output entire sequence, but instead the least likely symbol is displayed.'
    dummy[0] = train_signals[0] 
    output = net(np.vstack(( train_signals, dummy )))[-len(symbols):]

    for row in output:
        row[np.argmin(row)] = np.max(row) + 1

    print index_to_text(output, dictionary)
