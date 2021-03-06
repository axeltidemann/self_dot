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


duration_item = []
def updateWordDur(sentence):
    #global duration_item
    for word in sentence:
        duration_item.append((len(word), word))
    duration_item = list(set(duration_item))
    duration_item.sort()
    
# give each word a score for how much it belongs in the beginning of a sentence or in the end of a sentence   
# The resulting list here just gives the position (= the score for belonging in the end of a sentence).
# The score for membership in the beginning of a sentence is simply (1-endscore)
sentencePosition_item = []
def updatePositionMembership(sentence):
    #global sentencePosition_item
    lenSent = len(sentence)
    for i in range(lenSent):
        position = int((i/float(lenSent-1))*10)/10.0
        sentencePosition_item.append((position,sentence[i]))
    sentencePosition_item = list(set(sentencePosition_item))
    sentencePosition_item.sort()


def importFromFile(filename, useSavedAnalysis=1):
    if useSavedAnalysis and (os.path.isfile(filename+'_1save_words.sav')):
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
                updateWordDur(sentence)
                updatePositionMembership(sentence)
        saveToFile(filename)

def saveToFile(filename):
    pickle.dump(words, open(filename+'_1save_words.sav', 'wb'))
    pickle.dump(wordsInSentence, open(filename+'_1save_wordsInSentence.sav', 'wb'))
    pickle.dump(similarWords, open(filename+'_1save_similarWords.sav', 'wb'))
    pickle.dump(neighbors, open(filename+'_1save_neighbors.sav', 'wb'))
    pickle.dump(neighborAfter, open(filename+'_1save_neighborAfter.sav', 'wb'))
    pickle.dump(time_word, open(filename+'_1save_time_word.sav', 'wb'))
    pickle.dump(wordTime, open(filename+'_1save_wordTime.sav', 'wb'))
    pickle.dump(duration_item, open(filename+'_1save_duration_item.sav', 'wb'))
    pickle.dump(sentencePosition_item, open(filename+'_1save_sentencePosition_item.sav', 'wb'))

def loadFromFile(filename):
    global words, wordsInSentence, similarWords, neighbors, neighborAfter, time_word, wordTime, duration_item,sentencePosition_item
    words = pickle.load(open(filename+'_1save_words.sav', 'rb'))
    wordsInSentence = pickle.load(open(filename+'_1save_wordsInSentence.sav', 'rb'))
    similarWords = pickle.load(open(filename+'_1save_similarWords.sav', 'rb'))
    neighbors = pickle.load(open(filename+'_1save_neighbors.sav', 'rb'))
    neighborAfter = pickle.load(open(filename+'_1save_neighborAfter.sav', 'rb'))
    time_word = pickle.load(open(filename+'_1save_time_word.sav', 'rb'))
    wordTime = pickle.load(open(filename+'_1save_wordTime.sav', 'rb'))
    duration_item = pickle.load(open(filename+'_1save_duration_item.sav', 'rb'))
    sentencePosition_item = pickle.load(open(filename+'_1save_sentencePosition_item.sav', 'rb'))

            
if __name__ == '__main__':
    timeThen = time.time()    
    importFromFile('minimal_db.txt', 1)#association_test_db_short.txt', 0)#minimal_db.txt', 0) #association_test_db_full.txt')#
    print sentencePosition_item
    print time_word
    print 'processing time: %.1f ms'%((time.time()-timeThen)*1000)
