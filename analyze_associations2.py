#!/usr/bin/python
# -*- coding: latin-1 -*-

'''
A word is a classified segment (audio clip), as part of a sentence (audio file).
Classification of segments can be ambigous and can be updated/changed at a later time (dream state).

The audio_id in brain.wavs corresponds exactly to the semantic unit 'word'

allWords = list of all unique words (after classification)
wordTime = time when each word was recorded
timeWord = word recorded at a specific time
durationWord = duration of word
similarWords = words that sound similar but are not classified as the same
neighbors = words used immediately before or after this word
neighborsAfter = words used immediately after this word
wordFace = who has said this word (may be several faces)
faceWord = all words said by this face
'''


import re 
import re
findfloat=re.compile(r"[0-9.]*")

wavs = []               # list of wave files for each word (audio id) [[w1,w2,w3],[w4],...]
allWords = []           # just a list of all current audio_id (the indices for wavs)
wordTime = {}           # {id1:[time1, time2, t3...]], id2: ...}
timeWord = []           # [[time,id1],[time,id2],...]
durationWord = []       # [[dur, id1], [dur,id2]
similarWords = {}       # {id1: [[similar_id1, distance], [sim_id2, d], [sim_id3,d]], id2: [...]}
neighbors = {}          # {id1: [[neighb_id1, how_many_times], [neighb_id2, how_many_times]...], id2:[[n,h],[n,h]...]}
neighborAfter = {}      # as above, but only including words that immediately follow this word (not immediately preceding)
wordFace = {}           # {id1:[face1,face2,...], id2:[...]}
faceWord ={}            # [face1:[id1,id2...], face2:[...]}

def analyze(audio_id):
    print '*** association analysis comes (soon)'
    

def parseMarkerFile(markerfile):
    f = open(markerfile, 'r')
    segments = []
    enable = 0
    totaldur = 0
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        start = float(line[30:])
        if 'Total duration:'  in line: 
            enable = 0
            for item in findfloat.findall(line):
                if item != [] : totaldur = float(item)
        if enable:
            segments.append(float(line)+start) 
        if 'Sub segment start times:' in line: enable = 1
    #return segments
    return totaldur



if __name__ == '__main__':
    analyze()
