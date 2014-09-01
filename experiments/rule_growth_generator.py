#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Generate new sentences based on RuleGrowth rules

@author: Oeyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL

'''

from __future__ import print_function
import re
import random
#from subprocess import call
name_re = re.compile(r'\**\*:') 

def makeDict(rulefile):
    '''
    Make a dictionary where the keys are all left itemsets, 
    but we simplify itemsets containing a name so that the key is *only* the name.
    The rationale is that we want to look up a word, looking for a continuation of a sentence,
    and we might want to look up a name to get possible continuations.
    '''
    d = {}
    txt = open(rulefile)
    for line in txt.readlines():
        arrow = line.find('==>')
        support = line.find('#SUP')
        key = line[:arrow-1].split(',')
        isname = 0
        for item in key:
            if name_re.findall(item):
                key = tuple([item])
                isname = 1
        if isname == 0:
            key = tuple(key)
        value = line[arrow+4:support-1].split(',')
        isname = 0
        for item in value:    
            if name_re.findall(item):
                value = tuple([item])
                isname = 1
        if isname == 0:
            value = tuple(value)
        d.setdefault(key, []).append(value)
        d[key] = list(set(d[key]))
    return d



def generateNext(word, ruleDict):
    '''
    the generator function should take a word, use as a key to the ruleDict,
    then select an item from the dict[key] value
    if the item is a word or word sequence, use it
    if the item is a name, use that name as key to the dict, and use the resulting value (words)
    '''
    if word not in ruleDict.keys():
        print('no key', word, 'starting over')
        word = random.choice(ruleDict.keys())
        print('... with', word)
        return word
    candidates = ruleDict[word]
    new = random.choice(candidates)
    print('new', new)
    if name_re.findall(new[0]):
        print('found name', new, 'so trying again')
        new = generateNext(new, ruleDict)
    else:
        return new
        
if __name__ == '__main__':
    db = 'association_test_db_short.txt'
    ruleDict = makeDict(db+'-output-rules-inverted')
    dictfile= open(db+'-dict', 'w+')
    for k,v in ruleDict.iteritems():
        dictfile.write("{}: {}\n".format(k,v))
    sentence = []
    new = random.choice(ruleDict.keys())
    sentence.append(new)
    print('sentence is', sentence)
    for i in range(10):
        new = generateNext(new, ruleDict)
        sentence.append(new)
        print('sentence is', sentence)
