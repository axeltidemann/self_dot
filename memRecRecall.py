#!/usr/bin/python
# -*- coding: latin-1 -*-

# Python code to be run inside Csound, 
# reading soundfiles from memoryRecord into Csound,
# reading the corresponging text files containing transient markers into csound tables.
# The file also contain rudimentary (random) segment selection techniques, 
# as mock-up replacement for association rules.

import os
import random

files = os.listdir('../memoryRecording')

wavfiles = []
txtfiles = []

# find wav and txt files
for f in files:
    if (f.find('.wav') > -1):
        wavfiles.append(f)
    if (f.find('.txt') > -1):
        txtfiles.append(f)

def loadAudioAndMarker(basename):
    
    
