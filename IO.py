import os

import cv2
import numpy as np
import myCsoundAudioOptions

def video(state, camera, projector):
    print 'VIDEO PID', os.getpid()
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    video_feed = cv2.VideoCapture(0)
    frame_size = (160, 90)
    
    while True:
        _, frame = video_feed.read()
        frame = cv2.resize(frame, frame_size)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if state['record']:
            camera.append(np.ndarray.flatten(gray_image)/255.)

        try:
            cv2.imshow('Output', cv2.resize(np.reshape(projector.popleft(),
                                                       frame_size[::-1]), (640,360)))
        except:
            cv2.imshow('Output', np.random.rand(360,640))

        cv2.waitKey(100)

def audio(state, mic, speaker):
    print 'AUDIO PID', os.getpid()
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
    
    #fft_audio_in1 = csnd6.PVSDATEXT()
    
    fft_size = 256
    fftin_amptab = 1
    fftin_freqtab = 2
    fftout_amptab = 3
    fftout_freqtab = 4
    fftresyn_amptab = 11
    fftresyn_freqtab = 12
    fftin_amps = np.zeros(fft_size)
    fftin_freqs = np.zeros(fft_size)
    fftout_amps = np.zeros(fft_size)
    fftout_freqs = np.zeros(fft_size)
    
    while not stopflag:
        stopflag = cs.PerformKsmps()

        #test1 = cs.PvsoutGet(fft_audio_in1, "0")
        for i in range(fft_size):
            fftin_amps[i] = cs.TableGet(fftin_amptab, i)
            fftin_freqs[i] = cs.TableGet(fftin_freqtab, i)
            #fftout_amps[i] = cs.TableGet(fftout_amptab, i)
            #fftout_freqs[i] = cs.TableGet(fftout_freqtab, i)
            # clean test of unmodified fft resynthesis
            #cs.TableSet(fftresyn_amptab, i, fftin_amps[i])
            #cs.TableSet(fftresyn_freqtab, i, fftin_freqs[i])
            
        # get Csound channel data
        audioStatus = cs.GetChannel("audioStatus")

        if state['playfile']:
            print '[self.] wants to play {}'.format(state['playfile'])
            print '{}'.format(state['playfile'])
            cs.InputMessage('i3 0 5 "%s"'%'{}'.format(state['playfile']))
            state['playfile'] = False
        
        if state['record']:
            mic.append([cs.GetChannel("level1"), 
                        cs.GetChannel("envelope1"), 
                        cs.GetChannel("pitch1ptrack"), 
                        cs.GetChannel("pitch1pll"), 
                        cs.GetChannel("centroid1"),
                        cs.GetChannel("autocorr1"), 
                        cs.GetChannel("spread1"), 
                        cs.GetChannel("skewness1"), 
                        cs.GetChannel("kurtosis1"), 
                        cs.GetChannel("flatness1"), 
                        cs.GetChannel("crest1"), 
                        cs.GetChannel("flux1"), 
                        cs.GetChannel("epochSig1"), 
                        cs.GetChannel("epochRms1"), 
                        cs.GetChannel("epochZCcps1")])
            # Her: jeg aner hvorfor det ikke funker, men ser ikke akkurat naa hvordan jeg skal faa appendet til lista som lages like ovenfor her.
            #for i in range(fft_size):
            #    mic.append(fftin_amps[i])
            #    mic.append(fftin_freqs[i])
            '''
            To analyze self's own output and compare to it's input, do comparision of fftin (_amps, _freqs) with fftout (_amps, _freqs)
            '''

        try:
            sound = speaker.popleft()
            cs.SetChannel("respondLevel1", sound[0])
            cs.SetChannel("respondEnvelope1", sound[1])
            cs.SetChannel("respondPitch1ptrack", sound[2])
            cs.SetChannel("respondPitch1pll", sound[3])
            cs.SetChannel("respondCentroid1", sound[4])
            # test partikkel generator
            cs.SetChannel("partikkel1_amp", sound[0])
            cs.SetChannel("partikkel1_grainrate", sound[2])
            cs.SetChannel("partikkel1_wavfreq", sound[4])
            
            '''
            # partikkel parameters ready to be set
            partikkelparmOffset = 5
            cs.SetChannel("partikkel1_amp",sound[partikkelparmOffset+0])
            cs.SetChannel("partikkel1_grainrate",sound[partikkelparmOffset+1])
            cs.SetChannel("partikkel1_graindur",sound[partikkelparmOffset+2])
            cs.SetChannel("partikkel1_sustain",sound[partikkelparmOffset+3])
            cs.SetChannel("partikkel1_adratio",sound[partikkelparmOffset+4])
            cs.SetChannel("partikkel1_wavfreq",sound[partikkelparmOffset+5])
            cs.SetChannel("partikkel1_octaviation",sound[partikkelparmOffset+6])
            cs.SetChannel("partikkel1_async_amount",sound[partikkelparmOffset+7])
            cs.SetChannel("partikkel1_distribution",sound[partikkelparmOffset+8])
            cs.SetChannel("partikkel1_randomask",sound[partikkelparmOffset+9])
            cs.SetChannel("partikkel1_grFmFreq",sound[partikkelparmOffset+10])
            cs.SetChannel("partikkel1_grFmIndex",sound[partikkelparmOffset+11])
            cs.SetChannel("partikkel1_wavekey1",sound[partikkelparmOffset+12])
            cs.SetChannel("partikkel1_wavekey2",sound[partikkelparmOffset+13])
            cs.SetChannel("partikkel1_wavekey3",sound[partikkelparmOffset+14])
            cs.SetChannel("partikkel1_wavekey4",sound[partikkelparmOffset+15])
            cs.SetChannel("partikkel1_pitchFmFreq",sound[partikkelparmOffset+16])
            cs.SetChannel("partikkel1_pitchFmIndex",sound[partikkelparmOffset+17])
            cs.SetChannel("partikkel1_trainPartials",sound[partikkelparmOffset+18])
            cs.SetChannel("partikkel1_trainChroma",sound[partikkelparmOffset+19])
            cs.SetChannel("partikkel1_wavemorf",sound[partikkelparmOffset+20])
            '''
            '''
            # spectral parameters ready to be set
            spectralparmOffset = 25
            for i in range(fft_size):
                cs.TableSet(fftresyn_amptab, i, sound[spectralparmOffset+(i*2)])
                cs.TableSet(fftresyn_freqtab, i, sound[spectralparmOffset+(i*2)+1])
            '''                            
        except:
            cs.SetChannel("respondLevel1", 0)
            cs.SetChannel("respondEnvelope1", 0)
            cs.SetChannel("respondPitch1ptrack", 0)
            cs.SetChannel("respondPitch1pll", 0)
            cs.SetChannel("respondCentroid1", 0)
            # partikkel test
            cs.SetChannel("partikkel1_amp", 0)
