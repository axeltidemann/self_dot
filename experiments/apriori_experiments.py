#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with apriori
'''

import apriori
import random

support = 0.5
dataset = apriori.load_dataset()
print dataset
C1 = apriori.createC1(dataset)
print 'C1', C1
D = map(set,dataset)
print 'D', D
L1, support_data = apriori.scanD(D,C1,support)
print 'L1', L1
print 'support_data', support_data
k_length = 2
transactions = apriori.aprioriGen(L1, k_length)
print 'transactions', transactions
print '\n*** *** ***'
L,support_data = apriori.apriori(dataset, support)
print 'L', L
print 'support_data', support_data
rules = apriori.generateRules(L, support_data, min_confidence=0.7)
print 'rules', rules

ruleDict = apriori.generateRuleDict(rules)
print 'ruleDict', ruleDict

## testing
if __name__ == '__main__':
    predicate = random.choice(ruleDict.keys())
    predicate = list(predicate)[0]
    print 'predicate', predicate
    for i in range(4):
        sentence, predicate = apriori.generate([predicate], ruleDict)
        print 'the next item is', predicate
        print 'the current sentence is:', sentence