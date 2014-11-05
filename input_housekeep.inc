;******************************
; exit from Csound
        instr 9999
        exitnow
        endin

;******************************
; init chn values 
	instr 1
#include "chn_init.inc"
        endin
        
;******************************
; audio file input 
	instr 3

	Ssound	strget p4
	Smess	= "Playing back sound file at (microphone) input, the file plyed is: "
	Smess	strcat Smess, Ssound
		puts Smess, 1
;	Spath	="testsounds/"
;	S1	strcat Spath, Ssound
        ilen    filelen Ssound
        if p3 == 0 then
        p3 = ilen
        endif
	a1	soundin Ssound
	a2	= 0
		;outs a1, a2 ; monitor the file playback so we know it has been played
		chnmix a1, "in1"
		chnmix a2, "in2"
	endin

;******************************
; live audio input
	instr 4
	a1,a2	    inch 1,2
		        chnmix a1, "in1clean"
		        chnmix a2, "in2clean"

	aref		chnget "MyOutput"
    aenv        follow2 aref, 0.001, 0.9    ; amp of output
    kenv        downsamp aenv
    kgate       init 1
    kgate       = (dbfsamp(kenv)>-40?0:kgate) ; hysteresis gate, turn off when signal is over
    kgate       = (dbfsamp(kenv)<-60?1:kgate) ; complement of above
    agate       upsamp 1-kgate              ; upsample and invert, input conditioning for follow2
    aenv2       follow2 agate, 0.001, 0.8   ; follow2 add release fade out
    kduck       chnget "selfDucking"
    aenv2       = (aenv2*kduck)             ; user control for activation
    aenv2       = 1-aenv2    ; invert (normally at 1, duck to zero, then fade back in)
   
/*
;alternative ducking
	aref		chnget "MyOutput"
    kduck       chnget "selfDucking"
    aref        = aref * kduck
    aenv        follow2 aref, 0.001, 1.5    ; amp of output
    aenv        limit aenv, 0, 1
    aenv1	follow2 aenv*aref, 0.001, 1.2
    aenv2       = 1-aenv1    ; invert (normally at 1, duck to zero, then fade back in)
*/   

kdebug	chnget "printInputRms"
if kdebug > 0 then
krms	rms a1+a2
printk2 krms
krmsduck rms (a1*aenv2)+(a2*aenv2)
printk2 krmsduck, 5
endif

		        chnmix a1*aenv2, "in1"
		        chnmix a2*aenv2, "in2"

;		        chnmix a1, "in1"
;		        chnmix a2, "in2"
	endin

;******************************
; test tone
	instr 6
print p1, p2, p3, p4
	a1	oscili p4, 440, giSine
;		chnmix a1, "MasterOut1"
;		chnmix a1, "MasterOut2"
    outs a1, a1
	endin

;******************************
; find stereo position and merge mic 1 and 2
	instr 11
#include "panalyzer.inc"
        ain1    chnget "in1"
        ain2    chnget "in2"
		        chnset ain1+ain2, "in1"
	endin

;******************************
; measure roundtrip latency
	instr 12
#include "getLatency.inc"
	endin

;******************************
; get audio input noise print
	instr 13
#include "getAudioNoiseprint.inc"
	endin

;******************************
; suppress my own output by subtracting the output spectrum from the input
; and use noise print to remove static background noise
	instr 14
#include "suppressSelfnoise.inc"
	endin

;******************************
; get audio input noise floor (after noiseprint suppression if it is enabled)
	instr 15
print p1, p2, p3
#include "getAudioNoiseFloor.inc"
	endin

;******************************
; set audio input noise gate
	instr 16
	irms_dB		chnget "inputNoisefloor"
	irms		= ampdbfs(irms_dB)*0.9
	isize		= 8192
	iknee		= isize*0.2
	giNoiseFloor	ftgen 0, 0, isize, 7, 0, (isize*irms), 0, iknee, 1, (isize*(1-irms))-iknee, 1, 2, 1
	endin

;******************************
; apply noise gate
	instr 17
	a1		    chnget "in1"
	krms		rms a1
	krms		= krms * 1.7
	kgate		tablei krms, giNoiseFloor, 1
	kgate       tonek kgate, 5
	a1		    = a1 * kgate
			    chnset a1, "in1"
	endin

;******************************
; audio input mute
	instr 21
	ilevel	= p4
		chnset ilevel, "inputLevel" 
	endin

;******************************
; generic chn value setter
	instr 22
	Schan	strget p4
	ival	= p5
		chnset ival, Schan 
	endin

;******************************
; print chn value
	instr 23
	Schan	strget p4
	ival	chnget Schan
	print ival 
	endin

