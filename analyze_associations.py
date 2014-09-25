#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import zmq
import IO
import time

def analyze(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    assoc = context.socket(zmq.SUB)
    assoc.connect('tcp://{}:{}'.format(host, IO.ASSOCIATIONS))
    assoc.setsockopt(zmq.SUBSCRIBE, b'')

    poller = zmq.Poller()
    poller.register(assoc, zmq.POLLIN)

    timeStamp = time.time()
    while True:
    	#print 'assoc is running %i', time.time()-timeStamp
    	time.sleep(.1)
        events = dict(poller.poll(timeout=0))
        if assoc in events:
            markerfile = assoc.recv_json()
            segments = parseFile(markerfile)
            '''
            Do segment analysis, as in loadDbWeightedSet.
            This requires classification of recorded segments into recognized words
            '''

def parseFile(markerfile):
    f = open(markerfile, 'r')
    segments = []
    enable = 0
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        start = float(line[30:])
        if 'Total duration:'  in line: enable = 0
        if enable:
            segments.append(float(line)+start) 
        if 'Sub segment start times:' in line: enable = 1
    return segments


'''
A word is a classified segment (audio clip), as part of a sentence (audio file).
Classification of segments can be ambigous and can be updated/changed at a later time (dream state).
Here, we use 'word' to describe the semantic unit after classification,
and 'segment' as the raw input audio segment.

allSegments = sequential list of all words recorded
allWords = list of all unique words (after classification)
timeWhenUsed = time when word was recorded
duration = duration of word
positionInSentence = fractional value representing the position of word in sentence
wordsInSentence = words that have been used in the same sentence
similarWords = words that sound similar but are not classified as the same
neighbors = words used immediately before or after this word
'''


if __name__ == '__main__':
    analyze('localhost')
