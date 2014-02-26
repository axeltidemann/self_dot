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
stopflag = 0
fft_audio_in1 = numpy.zeros(1024)
fft_audio_in2 = numpy.zeros(1024)
offset = 0
while not stopflag:
    stopflag = cs.PerformKsmps()
    offset += 0.01
    offset %= 200
    cs.SetChannel("freq_offset", offset)
    test1 = cs.GetPvsChannel(fft_audio_in1, 0)
    test2 = cs.GetPvsChannel(fft_audio_in2, 1)
    cps = cs.GetChannel("cps")
print 'effective frequency in cps is %i'%cps
cs.Reset()

print 'exiting...'
time.sleep(1)

