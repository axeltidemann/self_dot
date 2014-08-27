#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with apriori
'''

import apriori
import random
import loadText

support = 0.03
loadText.importFromFile('association_test_db_full.txt')
dataset = loadText.rawPriori
#print dataset
C1 = apriori.createC1(dataset)
#print 'C1', C1
D = map(set,dataset)
#print 'D', D
L1, support_data = apriori.scanD(D,C1,support)
#print 'L1', L1
#print 'support_data', support_data
k_length = 2
transactions = apriori.aprioriGen(L1, k_length)
#print 'transactions', transactions
#print '\n*** *** ***'
L,support_data = apriori.apriori(dataset, support)
#print 'L', L
#print 'support_data', support_data
rules = apriori.generateRules(L, support_data, min_confidence=0.7)
#print 'rules', rules

ruleDict = apriori.generateRuleDict(rules)
print 'ruleDict', ruleDict
print '*** *** ***'
print 'keys', ruleDict.keys()
print '*** *** ***'

## testing
if __name__ == '__main__':
    predicate = random.choice(ruleDict.keys())
    sentence = list(predicate)[0]
    print 'predicate:', sentence
    for i in range(4):
        sentence, predicate = apriori.generate(sentence, ruleDict)
        sentence.append(predicate)    
        print 'next item:', predicate
        print 'the current sentence is:', sentence
        