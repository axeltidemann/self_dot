;******************************
; audio file input 
	instr 3

	Ssound	strget p4
;	Spath	="testsounds/"
;	S1	strcat Spath, Ssound
	a1	soundin Ssound
	a2	= 0
		outs a1, a2
		chnmix a1, "in1"
		chnmix a2, "in2"
	endin

;******************************
; live audio input
	instr 4
	a1,a2	inch 1,2
		chnmix a1, "in1"
		chnmix a2, "in2"
	endin

;******************************
; TODO
; 11 find stereo position and merge mic 1 and 2
	instr 11
	a1	chnget "in1"
	a2	chnget "in2"
		chnset a1+a2, "in1"
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
#include "getAudioNoiseFloor.inc"
	endin

;******************************
; set audio input noise gate
	instr 16
	irms_dB		chnget "inputNoisefloor"
	irms		= ampdbfs(irms_dB)
	isize		= 8192
	iknee		= isize*0.1
	giNoiseFloor	ftgen 0, 0, isize, 7, 0, (isize*irms), 0, iknee, 1, (isize*(1-irms))-iknee, 1
	endin

;******************************
; apply noise gate
	instr 17
	a1		chnget "in1"
	krms		rms a1
	krms		= krms * 1.7
	kgate		table krms, giNoiseFloor, 1
	a1		= a1 * kgate
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

