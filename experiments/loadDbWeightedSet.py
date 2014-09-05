#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Load text to database for association rules, for [self.]

Note:
updateSimilarWords(words) should be run again as maintenance (dream state), as it is only partially complete in realtime.
This is because it will only measure similarity to items already recorded.

@author: Oeyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL

'''
import re
import copy
import time
import difflib
import cPickle as pickle
import os.path

n=re.compile(r"\*.*\*:") # find name
w=re.compile(r"\b[0-9a-zA-Z']*\b") # find whole words
name = 'Anonymous'
words = set()

wordsInSentence = {}
def updateWordsInSentence(sentence):
    a = 0
    for word in sentence:
        v = wordsInSentence.setdefault(word, [])
        b = 0
        for other in sentence:
            if a != b: #don't count itself as an occurence
                otherInSentence = 0
                for word_score in v:
                    if other == word_score[0]:
                        word_score[1] += 1
                        otherInSentence = 1
                if otherInSentence == 0:
                    v.append([other, 1])
            b += 1
        a += 1

def similar(seq1, seq2):
    return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio()
                
similarWords = {}
def updateSimilarWords(sentence):
    words.update(sentence)
    for word in sentence:
        v = similarWords.setdefault(word, [])
        for other in words:
            if word != other: #don't count itself as an occurence
                if other not in [v[i][0] for i in range(len(v))]: #don't append if we already registered this word
                    v.append([other, similar(word, other)])
                
neighbors = {}
def updateNeighbors(sentence):    
    if len(sentence) == 1:
        return
    for i in range(len(sentence)):
        v = neighbors.setdefault(sentence[i], [])
        if i == 0: neighborPos = [i+1]
        elif i == len(sentence)-1: neighborPos = [i-1] 
        else: neighborPos = [i-1,i+1]
        for n in neighborPos:
            alreadyHere = 0
            for word_score in v:
                if sentence[n] == word_score[0]:
                    word_score[1] += 1
                    alreadyHere = 1
            if alreadyHere == 0:
                v.append([sentence[n], 1])

neighborAfter = {}
def updateNeighborAfter(sentence): 
    if len(sentence) == 1:
        return
    for i in range(len(sentence)-1):
        v = neighborAfter.setdefault(sentence[i], [])
        if sentence[i+1] in [v[j][0] for j in range(len(v))]:
            for word_score in v:
                if sentence[i+1] == word_score[0]:
                    word_score[1] += 1
        else:
            v.append([sentence[i+1], 1])


lettercount = 0
time_word = []
wordTime = {}
def updateTimeLists(sentence):
    global lettercount
    for word in sentence:
        lettercount += len(word)
        curTime = lettercount/20.0 # 20 letters per second "read speed"
        time_word.append((curTime, word))
        wordTime.setdefault(word, []).append(curTime)
        
def importFromFile(filename, useSavedAnalysis=1):
    if useSavedAnalysis and (os.path.isfile(filename+'_1save_words')):
        print 'using saved analysis'
        loadFromFile(filename)
    else:
        print 'analyzing...'
        f = open(filename, 'r')
        for line in f:
            sentence = []
            if len(n.findall(line)) > 0:
                name = w.findall(n.findall(line)[0])[0]
            else:
                wrds = w.findall(line)
            for item in wrds: 
                if item != '': sentence.append(item.lower())
            if len(sentence) > 0:
                words.update(sentence)
                updateNeighbors(sentence)
                updateNeighborAfter(sentence)
                updateWordsInSentence(sentence)
                updateSimilarWords(sentence)
                updateTimeLists(sentence)
        saveToFile(filename)

def saveToFile(filename):
    pickle.dump(words, open(filename+'_1save_words', 'wb'))
    pickle.dump(wordsInSentence, open(filename+'_1save_wordsInSentence', 'wb'))
    pickle.dump(similarWords, open(filename+'_1save_similarWords', 'wb'))
    pickle.dump(neighbors, open(filename+'_1save_neighbors', 'wb'))
    pickle.dump(neighborAfter, open(filename+'_1save_neighborAfter', 'wb'))
    pickle.dump(time_word, open(filename+'_1save_time_word', 'wb'))
    pickle.dump(wordTime, open(filename+'_1save_wordTime', 'wb'))

def loadFromFile(filename):
    global words, wordsInSentence, similarWords, neighbors, neighborAfter, time_word, wordTime
    words = pickle.load(open(filename+'_1save_words', 'rb'))
    wordsInSentence = pickle.load(open(filename+'_1save_wordsInSentence', 'rb'))
    similarWords = pickle.load(open(filename+'_1save_similarWords', 'rb'))
    neighbors = pickle.load(open(filename+'_1save_neighbors', 'rb'))
    neighborAfter = pickle.load(open(filename+'_1save_neighborAfter', 'rb'))
    time_word = pickle.load(open(filename+'_1save_time_word', 'rb'))
    wordTime = pickle.load(open(filename+'_1save_wordTime', 'rb'))
            
if __name__ == '__main__':
    timeThen = time.time()    
    importFromFile('association_test_db_short.txt', 0)#minimal_db.txt', 0) #association_test_db_full.txt')#
    print 'processing time: %.1f ms'%((time.time()-timeThen)*1000)
    #for word in words:
    #    print word
    #print wordsInSentence
    ''''
    print '\nneighbors'
    for k,v, in neighbors.iteritems():
        print k,v
    print '\nneighborAfter'
    for k,v, in neighborAfter.iteritems():
        print k,v
    '''
    print '\ntime_word'
    for item in time_word:
        print item
    print '\nwordTime'
    for k,v in wordTime.iteritems():
        print k,v
    '''
    updateSimilarWords(words) # this should be run again as maintenance (dream state), as it is only partially complete in realtime
    print '*** again ***'
    for k,v in similarWords.iteritems():
        print '***'
        print k,v
    '''