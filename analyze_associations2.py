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
import copy

wavsAsWords = []        # list of wave files for each word (audio id) [[w1,w2,w3],[w4],...]
allWords = set([])      # just a list of all current audio_id (the indices for wavs)
wordTime = {}           # {id1:[time1, time2, t3...]], id2: ...}
timeWord = []           # [[time,id1],[time,id2],...]
durationWord = []       # [[dur, id1], [dur,id2]
similarWords = {}       # {id1: [[similar_id1, distance], [sim_id2, d], [sim_id3,d]], id2: [...]}
neighbors = {}          # {id1: [[neighb_id1, how_many_times], [neighb_id2, how_many_times]...], id2:[[n,h],[n,h]...]}
neighborAfter = {}      # as above, but only including words that immediately follow this word (not immediately preceding)
wordFace = {}           # {id1:[face1,face2,...], id2:[...]}
faceWord ={}            # [face1:[id1,id2...], face2:[...]}

def analyze(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound):
    #print_us(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound)

    markerfile = filename[:-4]+'.txt'
    startTime, totalDur, segments = parseFile(markerfile)

    wavsAsWords = copy.copy(wavs)
    allWords.update([audio_id])
    timeWord.append((startTime, audio_id))
    wordTime.setdefault(audio_id, []).append(startTime)
    durationWord.append((totalDur, audio_id))
    #similarWords =      
    #neighbors = 
    #neighborAfter = 
    wordFace = copy.copy(sound_to_face)
    faceWord = copy.copy(face_to_sound)
    
    
def parseFile(markerfile):
    f = open(markerfile, 'r')
    segments = []
    enable = 0
    startTime = 0
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        startTime = float(line[30:])
        if 'Total duration:'  in line: 
            enable = 0
            totalDur = float(line[16:])
        if enable:
            segments.append(float(line)+startTime) 
        if 'Sub segment start times:' in line: enable = 1
    return startTime, totalDur, segments


def print_us(filename,audio_id,wavs,audio_hammings,sound_to_face,face_to_sound):
    print '*** association analysis ***'
    print filename

    print '\n***' 
    print audio_id

    print '\n***' 
    print wavs
    
    print '\n***' 
    # would like to have distance from id to all other ids not in same class
    # or distance from this class to all other classes 
    print audio_hammings # distance from id to all other in same class?

    print '\n***' 
    print sound_to_face

    print '\n***' 
    print face_to_sound
    print '\n***' 
    




if __name__ == '__main__':
    analyze()
