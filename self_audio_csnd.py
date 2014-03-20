#!/usr/bin/python
# -*- coding: latin-1 -*-

''' 
Audio i/o, analysis, resynthesis. For [self.]

@author: Øyvind Brandtsegg
@contact: obrandts@gmail.com
@license: GPL
'''

import csnd6
import time
import numpy

cs = csnd6.Csound()
cs.Compile("self_audio_csnd.csd")
cs.Start()
stopflag = 0
#fft_audio_in1 = numpy.zeros(1024)
#fft_audio_in2 = numpy.zeros(1024)
offset = 0
while not stopflag:
    stopflag = cs.PerformKsmps()
    
    #offset += 0.01
    #offset %= 200
    #cs.SetChannel("freq_offset", offset)
    #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
    #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

    # get Csound channel data
    level1 = cs.GetChannel("level1")
    level2 = cs.GetChannel("level2")
    envelope1 = cs.GetChannel("envelope1")
    envelope2 = cs.GetChannel("envelope2")
    pitch1 = cs.GetChannel("pitch1")
    pitch2 = cs.GetChannel("pitch2")
    centr1 = cs.GetChannel("centroid1")
    centr2 = cs.GetChannel("centroid2")

    # self does its imitation magic and writes new values to Csound channels
    # for now just copying the input values to test the signal chain
        
    # set Csound channel data
    cs.SetChannel("imitateLevel1", level1)
    cs.SetChannel("imitateEnvelope1", envelope1)
    cs.SetChannel("imitatePitch1", pitch1)
    cs.SetChannel("imitateCentroid1", centr1)

cs.Reset()

print 'exiting...'
time.sleep(1)

