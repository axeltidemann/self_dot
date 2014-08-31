#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Generate new sentences based on RuleGrowth rules

@author: Oeyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL

'''

from __future__ import print_function
import re
from subprocess import call

def makeDict(rulefile):
    '''
    Make a dictionary where the keys are all left itemsets, 
    but we simplify itemsets containing a name so that the key is *only* the name.
    The rationale is that we want to look up a word, looking for a continuation of a sentence,
    and we might want to look up a name to get possible continuations.
    '''
    d = {}
    txt = open(rulefile)
    name_re = re.compile(r'\**\*:') 
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



#def generateNext(word):
    '''
    the generator function should take a word, use as a key to the ruleDict,
    then select an item from the dict[key] value
    if the item is a word or word sequence, use it
    if the item is a name, use that name as key to the dict, and use the resulting value (words)
    '''
    ##

if __name__ == '__main__':
    db = 'association_test_db_short.txt'
    ruleDict = makeDict(db+'-output-rules-inverted')
    dictfile= open(db+'-dict', 'w+')
    for k,v in ruleDict.iteritems():
        dictfile.write("{}: {}\n".format(k,v))
    
