#!/usr/bin/python
# -*- coding: latin-1 -*-

# Python code to be run from inside Csound, 
# reading soundfiles from memoryRecord into Csound,
# reading the corresponging text files containing transient markers into csound tables.
# The file also contain rudimentary (random) segment selection techniques, 
# as mock-up replacement for association rules.

import os
import csnd6
import random

csInstance = csnd6.csoundGetInstance(_CSOUND_)
memoryPath = '../memoryRecording/'
files = os.listdir(memoryPath)

def getBasenames():
    wavfiles = []
    txtfiles = []
    basenames = []
    # find wav and txt files
    for f in files:
        if (f.find('.wav') > -1):
            wavfiles.append(f)
        if (f.find('.txt') > -1):
            txtfiles.append(f)        
    # find base names for files that exist both with wav and txt extension
    for f in wavfiles:
        basename = f.split(".wav")[0]
        for t in txtfiles:
            if basename in t:
                basenames.append(basename)
    return basenames

def parseMarkerfile(basename):
    f = file(basename+".txt", 'r')
    markers = ''
    for line in f:
        try:
            num = float(line)
            markers += str(num)+' '
        except:
            pass    
        if "Total duration" in line:
            totaldur = float(line[15:])
    return markers, totaldur

def loadAudioAndMarkers(basename):
    markers, totaldur = parseMarkerfile(memoryPath+basename)
    csnd6.csoundInputMessage(csInstance, 'i 71 0 .1 \"%s.wav\" \"%s\" %f'%(memoryPath+basename, markers, totaldur))
    print 'cs event sent:', 'i 71 0 .1 \"%s.wav\" \"%s\" %f'%(basename, markers, totaldur)

def loadRandomFromMemory():
    loadAudioAndMarkers(random.choice(getBasenames()))

loadRandomFromMemory()
