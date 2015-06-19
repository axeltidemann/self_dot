#!/usr/bin/python
# -*- coding: latin-1 -*-

#    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
#
#    This file is part of [self.]
#
#    [self.] is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 
#    as published by the Free Software Foundation.
#
#    [self.] is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with [self.].  If not, see <http://www.gnu.org/licenses/>.

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''
from collections import deque
import sys
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

import myCsoundAudioOptions

if __name__ == '__main__':
    playfile = sys.argv[1]
    
    import csnd6
    cs = csnd6.Csound()
    arguments = csnd6.CsoundArgVList()
    arguments.Append("dummy")
    arguments.Append("self_dot.csd")
    csoundCommandline = myCsoundAudioOptions.myAudioDevices
    comlineParmsList = csoundCommandline.split(' ')
    for item in comlineParmsList:
        arguments.Append("%s"%item)
    cs.Compile(arguments.argc(), arguments.argv())
    stopflag = 0
    zeroChannelsOnNoBrain = 1
    
    fftsize = int(cs.GetChannel("fftsize"))
    ffttabsize = fftsize/2
    fftin_amptab = 1
    fftin_freqtab = 2
    fftout_amptab = 4
    fftout_freqtab = 5
    fftresyn_amptab = 7
    fftresyn_freqtab = 8
    
    # optimizations to avoid function lookup inside loop
    tGet = cs.TableGet 
    tSet = cs.TableSet
    cGet = cs.GetChannel
    cSet = cs.SetChannel
    perfKsmps = cs.PerformKsmps
    fftbinindices = range(ffttabsize)
    fftin_amptabs = [fftin_amptab]*ffttabsize
    fftin_freqtabs = [fftin_freqtab]*ffttabsize
    fftout_amptabs = [fftout_amptab]*ffttabsize
    fftout_freqtabs = [fftout_freqtab]*ffttabsize
    fftresyn_amptabs = [fftresyn_amptab]*ffttabsize
    fftresyn_freqtabs = [fftresyn_freqtab]*ffttabsize
    fftzeros = [0]*ffttabsize
    fftconst = [0.1]*ffttabsize
    fftin_amplist = [0]*ffttabsize
    fftin_freqlist = [0]*ffttabsize

    print 'Playing {}'.format(playfile)
    cs.InputMessage('i3 0 0 \"{}\"'.format(playfile))
    cs.InputMessage('i22 	0 .1 \"inputNoisefloor\" -30')
    audio_buffer = deque()
    
    while not stopflag:
        stopflag = perfKsmps()
        fftinFlag = cGet("pvsinflag")
        fftoutFlag = cGet("pvsoutflag")
        
        if fftinFlag:
            fftin_amplist = map(tGet,fftin_amptabs,fftbinindices)
            fftin_freqlist = map(tGet,fftin_freqtabs,fftbinindices)
            #bogusamp = map(tSet,fftresyn_amptabs,fftbinindices,fftin_amplist)
            #bogusfreq = map(tSet,fftresyn_freqtabs,fftbinindices,fftin_freqlist)
        if fftoutFlag:
            fftout_amplist = map(tGet,fftout_amptabs,fftbinindices)
            fftout_freqlist = map(tGet,fftout_freqtabs,fftbinindices)

        # get Csound channel data
        audioStatus = cGet("audioStatus")
        audioStatusTrig = cGet("audioStatusTrig")
        transient = cGet("transient")

        audio_buffer.append(np.array([cGet("level1"),
                                      cGet("pitch1ptrack"), 
                                      cGet("pitch1pll"), 
                                      cGet("autocorr1"), 
                                      cGet("centroid1"),
                                      cGet("spread1"), 
                                      cGet("skewness1"), 
                                      cGet("kurtosis1"), 
                                      cGet("flatness1"), 
                                      cGet("crest1"), 
                                      cGet("flux1"), 
                                      cGet("epochSig1"), 
                                      cGet("epochRms1"), 
                                      cGet("epochZCcps1")]))# + fftin_amplist + fftin_freqlist))

        if audioStatusTrig < 0:
            stopflag = True

    print 'Length of audio buffer: {}'.format(len(audio_buffer))
            
    analysis = np.array(list(audio_buffer))
    # scaler = pp.MinMaxScaler() 
    # scaled_audio = scaler.fit_transform(analysis)

    # fft_index = 14
    
    # plt.plot(scaled_audio[:,:fft_index])
    # plt.title('Without FFT')
    # plt.ylim([0,1])
    # plt.xlim([0,len(audio_buffer)])

    # plt.figure()
    # plt.plot(scaled_audio[:,fft_index:])
    # plt.title('FFT')
    # plt.ylim([0,1])
    # plt.xlim([0,len(audio_buffer)])
    
    # plt.show()

    filename = '{}.npy'.format(playfile)
    pickle.dump(analysis, open(filename, 'w'))
    print 'Saved numpy array as {}'.format(filename)
