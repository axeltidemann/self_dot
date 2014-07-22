; ******************************
; read audio segments from memoryRecording
        instr 70
        ; use python and ._CSOUND_ to get the file names and initiate events of instr 71
	; major parts of this instr (python defs) can be moved to global area

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

ftableNums = []
# use ftable nums between 800 and 899, with markers loaded in tables 900 to 999
filesloaded ={}
# populate filesloaded with sublist [filename, ftablenum, accesstime]
# where accesstime is initially set to load time
# accesstime is updated each time Csound makes use of the file
# When Csound needs to use the file, query Python for the ftable num and update accesstime.
# GC: Files that has not been used for N minutes (20 min?) are GC'ed, 
# ...make Csound instrument to ftfree the table and also update Python list of available table nums

def getftableNum():
    tableNum = 800
    if len(ftableNums)<99:
        while tableNum in ftableNums:
            tableNum += 1
        if tableNum > 899: tableNum = 800
        ftableNums.append(tableNum)
    return tableNum

def loadAudioAndMarkers(basename):
    if basename not in filesloaded.keys():
        markers, totaldur = parseMarkerfile(memoryPath+basename)
        ftableNum = getftableNum()
        csnd6.csoundInputMessage(csInstance, 'i 71 0 .1 %i \"%s.wav\" \"%s\" %f'%(ftableNum, memoryPath+basename, markers, totaldur))
        print 'cs event sent:', 'i 71 0 .1 \"%s.wav\" \"%s\" %f'%(basename, markers, totaldur)
        filesloaded[basename] = [ftableNum, time.time()]

def soundfileGC(gcPeriod=20):
    gcTime = time.time()-(gcPeriod*60)
    for f in filesloaded.keys():
        if filesloaded[f][1] < gcTime: 
            ftableNum = filesloaded[f][0]
            del filesloaded[f]
            soundfileFree(ftableNum)

def soundfileFree(ftableNum):
    ##FIX ME

def loadRandomFromMemory():
    loadAudioAndMarkers(random.choice(getBasenames()))

loadRandomFromMemory()
}}
        endin

        instr 71
; load sound file and corresonding transient markers
	itabNum		= p4
        Swavname        strget p5
        iSound          ftgen itabNum, 0, 0, 1, Swavname, 0, 0, 0

        Smarkers        strget p6
        iTotdur         = p7
        iMarkers        ftgen itabNum+100, 0, 1024, 7, -1, 1024, -1 ; we can use -1 as eof, the first address not explicitly set will now contain -1
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
