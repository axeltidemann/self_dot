#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
Experiments with apriori
'''

import apriori
import random
import loadText

support = 0.1
loadText.importFromFile('snowflakes_db.txt')
dataset = loadText.rawPriori
#print dataset
C1 = apriori.createC1(dataset)
#print 'C1', C1
D = map(set,dataset)
#print 'D', D
L1, support_data = apriori.scanD(D,C1,support)
#print 'L1', L1
#print 'support_data', support_data
print 'support_data'
for k,v in support_data.iteritems():
    print k,v
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

print 'ruleDict'
for k,v in ruleDict.iteritems():
    print '\t', k,  "".join(' ' for i in range(30-len(''.join(item for item in list(k)))-len(k)*4)), v
print '*** *** ***'

#print 'keys', ruleDict.keys()
#print '*** *** ***'


## testing
if __name__ == '__main__':
    #print '\n\n***\n'
    predicate = random.choice(ruleDict.keys())
    sentence = list(predicate)
    association = list(predicate)
    print 'predicate:', predicate
    for i in range(4):
        association, predicate = apriori.generate(association, ruleDict)  
        print '\t association:', predicate
        sentence.append(predicate)
        print 'the current sentence is:', sentence
