; ******************************
; read audio segments from memoryRecording
        instr 70
        ; use python and ._CSOUND_ to get the file names and initiate events of instr 71
        pyruni {{
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
}}
        endin

        instr 71
; load sound file and corresonding transient markers
        Swavname        strget p4
        iSound          ftgen 0, 0, 0, 1, Swavname, 0, 0, 0

        Smarkers        strget p5
        iTotdur         = p6
        iMarkers        ftgen 0, 0, 1024, 7, -1, 1024, -1 ; we can use -1 as eof, the first address not explicitly set will now contain -1
                        tableiw  iTotdur, 0, iMarkers
        index           = 1
        parseMarkers:
        ipos            strindex Smarkers, " "
        Snum            strsub Smarkers, 0, ipos
        Smarkers        strsub  Smarkers, ipos+1, -1        
        inum            strtod Snum
                        tableiw  inum, index, iMarkers
        index           += 1
        ilen            strlen Smarkers
        if ilen > 1 then
        igoto parseMarkers
        endif

        endin

; ******************************
; straight playback of audio segments
        instr 73
; FIXME
        endin

; ******************************
; partikkel instr 
	instr 74
#include "partikkel2_chn.inc"
#include "partikkel2_self.inc"
			chnmix aDry, "SecondaryDry"
			chnmix aDelaySend, "SecondaryDelaySend"
			chnmix aReverbSend, "SecondaryReverbSend"
	endin

; ******************************
; spectral synth playback of audio segments
        instr 75
; FIXME
        endin

; ******************************
; delay for secondary assoc 
	instr 77
	a1              chnget "SecondaryDelaySend"
        kTime		chnget "SecondaryDelayFeed"
        kFeed		chnget "SecondaryDelayFeed"
	kFiltFq		chnget "SecondaryDelayFiltFq"
        imaxdel         = 2
	adelTim		upsamp 	kTime				; smoothing
	adelTim		tone	adelTim, 2			; smoothing
	adummy		delayr imaxdel				; establish delay line
	aout		deltapi adelTim				; tap delay 
	adelw		= a1 + (aout*kFeed)			; mix input and feedback
	adelw		butterlp adelw, kFiltFq			; filter delay signal
			delayw	adelw				; write source to delay line
	aout		butterhp aout, 120
			chnmix aout, "SecondaryDry"
	endin

; ******************************
; reverb for secondary assoc 
	instr 78
	a1              chnget "SecondaryReverbSend"
        kFeed		chnget "SecondaryReverbFeed"
	kFiltFq		chnget "SecondaryReverbFiltFq"
	iPitchMod	= 0.9	
	aL,aR		reverbsc a1, a1, kFeed, kFiltFq, sr, iPitchMod
	aout		butterhp aL+aR, 120
			chnmix aout, "SecondaryDry"
	endin

; ******************************
; mixer for secondary assoc 
	instr 79
	klevel          chnget "SecondaryMasterLevel"
        klevel          = ampdbfs(klevel)
	a1              chnget "SecondaryDry"
        aout            = a1*klevel        
			chnmix aout, "MasterOut2"
        ; reset channels for secondary assoc
        a0              = 0
                        chnset a0, "SecondaryDry"
			chnset a0, "SecondaryDelaySend"
			chnset a0, "SecondaryReverbSend"

	endin
