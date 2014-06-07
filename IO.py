import cPickle as pickle
import glob
import multiprocessing as mp

import cv2
import numpy as np

from utils import sleep, filesize
import myCsoundAudioOptions
from AI import live
from communication import send

def video(state, camera, projector):
    me = mp.current_process()
    me.name = 'VIDEO'
    print me.name, 'PID', me.pid

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
    me = mp.current_process()
    me.name = 'AUDIO'
    print me.name, 'PID', me.pid

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
        
        if state['autolearn']:
            if audioStatusTrig > 0:
                state['record'] = True
            if audioStatusTrig < 0:
                state['record'] = False
                send('learn')

        if state['autorespond']:
            if audioStatusTrig > 0:
                state['record'] = True
            if audioStatusTrig < 0:
                state['record'] = False
                send('respond')

        if state['selfvoice']:
            mode = '{}'.format(state['selfvoice'])
            if mode in ['partikkel', 'spectral', 'noiseband']:
                print 'self change voice to...', mode
                cs.InputMessage('i -51 0 .1')
                cs.InputMessage('i -52 0 .1')
                cs.InputMessage('i -53 0 .1')
                if mode == 'noiseband': cs.InputMessage('i 51 0 -1')
                if mode == 'partikkel': cs.InputMessage('i 52 0 -1')
                if mode == 'spectral': cs.InputMessage('i 53 0 -1')
            else:
                print 'unknown voice mode', mode
            state['selfvoice'] = False
            
        if state['inputLevel']:
            mode = '{}'.format(state['inputLevel'])
            if mode == 'mute': cs.InputMessage('i 21 0 .1 0')
            if mode == 'unmute': cs.InputMessage('i 21 0 .1 1')
            if mode == 'reset': 
                cs.InputMessage('i 21 0 .1 0')
                cs.InputMessage('i 21 1 .1 1')
            state['inputLevel'] = False

        if state['calibrateAudio']:
            calibratePeriod = 2
            cs.InputMessage('i -17 0 1') # turn off old noise gate
            cs.InputMessage('i 14 0 %f'%calibratePeriod) # get level
            cs.InputMessage('i 15 %f 0.1'%(calibratePeriod+0.1)) # set noise gate shape
            cs.InputMessage('i 17 %f -1'%(calibratePeriod+0.2)) # turn on new noise gate
            state['calibrateAudio'] = False

        if state['csinstr']:
            # generic csound instr message
            cs.InputMessage('{}'.format(state['csinstr']))
            print 'sent {}'.format(state['csinstr'])
            state['csinstr'] = False
            
        if state['playfile']:
            print '[self.] wants to play {}'.format(state['playfile'])
            print '{}'.format(state['playfile'])
            cs.InputMessage('i3 0 5 "%s"'%'{}'.format(state['playfile']))
            state['playfile'] = False

        if state['record']:
            mic.append([cGet("level1"), 
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
                        cGet("epochZCcps1")] + fftin_amplist + fftin_freqlist)

        try:
            sound = speaker.popleft()
            cSet("respondLevel1", sound[0])
            cSet("respondPitch1ptrack", sound[1])
            cSet("respondPitch1pll", sound[2])
            cSet("respondCentroid1", sound[4])
            # test partikkel generator
            cSet("partikkel1_amp", sound[0])
            cSet("partikkel1_grainrate", sound[1])
            cSet("partikkel1_wavfreq", sound[4])
            cSet("partikkel1_graindur", sound[3]+0.1)
            # transfer fft frame
            bogusamp = map(tSet,fftresyn_amptabs,fftbinindices,sound[15:ffttabsize+15])
            bogusfreq = map(tSet,fftresyn_freqtabs,fftbinindices,sound[ffttabsize+15:ffttabsize+15+ffttabsize])
            
            '''
            # partikkel parameters ready to be set
            partikkelparmOffset = 5
            cSet("partikkel1_amp",sound[partikkelparmOffset+0])
            cSet("partikkel1_grainrate",sound[partikkelparmOffset+1])
            cSet("partikkel1_graindur",sound[partikkelparmOffset+2])
            cSet("partikkel1_sustain",sound[partikkelparmOffset+3])
            cSet("partikkel1_adratio",sound[partikkelparmOffset+4])
            cSet("partikkel1_wavfreq",sound[partikkelparmOffset+5])
            cSet("partikkel1_octaviation",sound[partikkelparmOffset+6])
            cSet("partikkel1_async_amount",sound[partikkelparmOffset+7])
            cSet("partikkel1_distribution",sound[partikkelparmOffset+8])
            cSet("partikkel1_randomask",sound[partikkelparmOffset+9])
            cSet("partikkel1_grFmFreq",sound[partikkelparmOffset+10])
            cSet("partikkel1_grFmIndex",sound[partikkelparmOffset+11])
            cSet("partikkel1_wavekey1",sound[partikkelparmOffset+12])
            cSet("partikkel1_wavekey2",sound[partikkelparmOffset+13])
            cSet("partikkel1_wavekey3",sound[partikkelparmOffset+14])
            cSet("partikkel1_wavekey4",sound[partikkelparmOffset+15])
            cSet("partikkel1_pitchFmFreq",sound[partikkelparmOffset+16])
            cSet("partikkel1_pitchFmIndex",sound[partikkelparmOffset+17])
            cSet("partikkel1_trainPartials",sound[partikkelparmOffset+18])
            cSet("partikkel1_trainChroma",sound[partikkelparmOffset+19])
            cSet("partikkel1_wavemorf",sound[partikkelparmOffset+20])
            '''
        except:
            cSet("respondLevel1", 0)
            cSet("respondPitch1ptrack", 0)
            cSet("respondPitch1pll", 0)
            cSet("respondCentroid1", 0)
            # partikkel test
            cSet("partikkel1_amp", 0)
            cSet("partikkel1_grainrate", 0)
            cSet("partikkel1_wavfreq", 0)
            # zero fft frame 
            bogusamp = map(tSet,fftresyn_amptabs,fftbinindices,fftzeros)

            
def load_cns(state, mic, speaker, camera, projector):

    for filename in glob.glob(state['load']+'*'):
        audio_net, audio_video_net, scaler = pickle.load(file(filename, 'r'))
        mp.Process(target=live, args=(state, mic, speaker, camera, projector, audio_net, audio_video_net, scaler)).start()
        print 'Brain loaded from file {} ({})'.format(filename, filesize(filename))

