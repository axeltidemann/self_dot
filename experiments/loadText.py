#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Load text to database for association rules, for [self.]

@author: Oeyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL

'''
import re
import copy

n=re.compile(r"\*.*\*:") # find name
w=re.compile(r"\b[0-9a-zA-Z']*\b") # find whole words
name = 'Anonymous'
rawPriori = []
lettercount = 0
words = {}
persons = {}
times = []

def organizeData(sentence):
    # some experiments at different way of organizing the data
    global lettercount # just a mock-up for "time of recording of word" for text input
    for word in sentence:
        sentenc1 = copy.copy(sentence)
        while word in sentenc1:
            sentenc1.remove(word)
        lettercount += len(word)
        curTime = lettercount/20.0 # 20 letters per second "read speed"
        words.setdefault(word, []).extend(sentenc1)
        persons.setdefault(name, []).append(word)
        times.append((curTime, word))

def importFromFile(filename):
    f = open(filename, 'r')
    for line in f:
        sentence = []
        if len(n.findall(line)) > 0:
            name = w.findall(n.findall(line)[0])[0]
        else:
            wrds = w.findall(line)
        for item in wrds: 
            if item != '': sentence.append(item)
        if len(sentence) > 0:
            rawPriori.append(sentence)
            organizeData(sentence)

if __name__ == '__main__':
    importFromFile('association_test_db_short.txt')
    for k,i in words.iteritems():
        print '*W*', k,i
    for k,i in persons.iteritems():
        print '*N*', k,i
    for t in times: 
        print '*T*', t 


    

