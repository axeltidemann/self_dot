#!/usr/bin/python
# -*- coding: latin-1 -*-

''' Finds the best parameters for the neural networks that performs audio recognition, using evolution.

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL
'''

import cPickle as pickle

import numpy as np
from pyevolve import G1DList, GSimpleGA, Mutators

from AI import _train_network

audio_memories = pickle.load(open('counts.pickle'))

test = audio_memories[1::2]
train = audio_memories[:-1:2]

idxs = [0,6,7,8,9,12]

new_test = [t[:,idxs] for t in test ]
new_train = [t[:,idxs] for t in train ]

targets = []

for i, memory in enumerate(train):
    target = np.zeros((memory.shape[0], len(train))) - 1
    target[:,i] = 1
    targets.append(target)

def fitness(chromosome):
    recognizer = _train_network(new_train, targets, output_dim=chromosome[0]*10, leak_rate=float(chromosome[1])/100, bias_scaling=float(chromosome[2])/100)
    return np.mean([ i == np.argmax(np.mean(recognizer(memory), axis=0)) for i, memory in enumerate(new_test) ])

genome = G1DList.G1DList(3) # net_size, leak_rate, bias_scaling
genome.evaluator.set(fitness)
genome.mutator.set(Mutators.G1DListMutatorIntegerRange)
genome.setParams(rangemin=10, rangemax=100)
ga = GSimpleGA.GSimpleGA(genome)
ga.setMultiProcessing(True)
ga.setMutationRate(.5)
ga.setPopulationSize(100)
ga.setGenerations(100)
ga.evolve(freq_stats=10)
print ga.bestIndividual()


