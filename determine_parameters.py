#!/usr/bin/python
# -*- coding: latin-1 -*-

#    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
#
#    This file is part of [self.]
#
#    [self.] is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 
#    as published by the Free Software Foundation.
#
#    [self.] is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with [self.].  If not, see <http://www.gnu.org/licenses/>.

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

''' Finds the best parameters for the neural networks that performs audio recognition, using evolution.'''

import cPickle as pickle

import numpy as np
from pyevolve import G1DList, GSimpleGA, Mutators

from AI import _train_network
from brain import chop

audio_memories = pickle.load(open('counts.pickle'))

test = audio_memories[1::2]
train = audio_memories[:-1:2]

idxs = [0,6,7,8,9,12]

new_test = [chop(t[:,idxs]) for t in test ]
new_train = [chop(t[:,idxs]) for t in train ]

targets = []

for i, memory in enumerate(new_train):
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


