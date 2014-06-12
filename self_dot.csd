<CsoundSynthesizer>
<CsOptions>
;-odac1 -iadc0 
-odac18 -iadc20
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 128
	nchnls = 2	
	0dbfs = 1

; pvs ftables
	gifftsize 	= 512
			chnset gifftsize, "fftsize"
	giFftTabSize	= (gifftsize / 2)+1
	gifna     	ftgen   1 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis
	gifnf     	ftgen   2 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis

	gifnaSelf     	ftgen   4 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of my own output
	gifnfSelf     	ftgen   5 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of my own output

	gifnaResyn     	ftgen   7 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs resynthesis
	gifnfResyn     	ftgen   8 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs resynthesis

	gifnaIn     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of input (self output suppression)
	gifnaOut     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of output (self output suppression)

	giNoiseFloor	ftgen 0, 0, 8192, 2, 0					; just init, to be used as noise gate 
	gifnaNoiseIn   	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of bacground noise
	gifnaNoise     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of bacground noise

; classic waveforms
	giSine		ftgen	0, 0, 65536, 10, 1					; sine wave
	giCosine	ftgen	0, 0, 8192, 9, 1, 1, 90					; cosine wave
	giTri		ftgen	0, 0, 8192, 7, 0, 2048, 1, 4096, -1, 2048, 0		; triangle wave 

; grain envelope tables
	giSigmoRise 	ftgen	0, 0, 8193, 19, 0.5, 1, 270, 1				; rising sigmoid
	giSigmoFall 	ftgen	0, 0, 8193, 19, 0.5, 1, 90, 1				; falling sigmoid
	giExpFall	ftgen	0, 0, 8193, 5, 1, 8193, 0.00001				; exponential decay
	giTriangleWin 	ftgen	0, 0, 8193, 7, 0, 4096, 1, 4096, 0			; triangular window 

#include "udos.inc"

;******************************
; audio file input 
	instr 3

	Ssound	strget p4
	Spath	="testsounds/"
	S1	strcat Spath, Ssound
	a1	soundin S1
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
	irms		chnget "inputNoisefloor"
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

;******************************
; analysis  of audio input 
	instr 31
	
	a1		chnget "in1"
	kinlevel	chnget "inputLevel"
	a1		= a1*kinlevel
	a0		= 0
			chnset a0, "in1"
			chnset a0, "in2"
	ifna		= gifna
	ifnf		= gifnf
#include "audio_analyze.inc"

; ***************
; write to chn
			chnset kflag, "pvsinflag"

	Stest		sprintfk "input audio status %i, trig status %i, dbstart %f", kstatus, kstatusTrig, kdBStart
			puts Stest, 2+(kstatus + kstatusTrig)

			chnset kstatus, "audioStatus"
			chnset kstatusTrig, "audioStatusTrig"
			chnset ktrig1, "transient"
			chnset krms1, "level1"
			chnset kcps1, "pitch1ptrack"
			chnset kcps1p, "pitch1pll"
			chnset kautocorr, "autocorr1"
			chnset kcentroid, "centroid1"
			chnset kspread, "spread1"
			chnset kskewness, "skewness1"
			chnset kurtosisM, "kurtosis1"
			chnset kflatness, "flatness1"
			chnset kcrest, "crest1"
			chnset kflux, "flux1"
			chnset kepochSig, "epochSig1"
			chnset kepochRms, "epochRms1"
			chnset kepochZCcps, "epochZCcps1"

	kinputMonitor	chnget "inputMonitor"
			chnmix a1*kinputMonitor, "MasterOut1"
			chnmix a1*kinputMonitor, "MasterOut2"

	endin

; ******************************
; generate test data, 
; get spectral profile from sound file, 
; to use as amp or freq data for selfvoice spectral synth
	instr 42
	Ssound	strget p4
	Spath	="testsounds/"
	S1	strcat Spath, Ssound
	a1	soundin S1
	a2	= 0

	ifna	= gifnaResyn
	ifnf	= gifnfResyn
#include "audio_analyze.inc"

		outs a1, a2
	endin

; ************************************************************
; resynthesis/imitation

; ******************************
; basic subtractive harmonic synth
	instr 51			

	krms1 		chnget "respondLevel1"
	;kcps1 		chnget "respondPitch1ptrack"
	kcps1 		chnget "respondPitch1pll"
	kcentro1 	chnget "respondCentroid1"

	krms1		limit krms1, 0, 1
	krms1		mediank krms1, 100, 100
	krms1		tonek krms1, 20

	kcps1		limit kcps1, 20, 2000
	kcps1		tonek kcps1, 20
	kcentro1	limit kcentro1, 20, 2000
	kcentro1	tonek kcentro1, 20

	; simple subtractive synth, creating N harmonic bands from pitch info,
	; using centroid to further filter which of the harmonics are more prominent
	anoise		rnd31 1,1
	afilt1a		butterbp anoise, kcps1, kcps1*0.05
	afilt1b		butterbp anoise, kcps1*2, kcps1*0.05
	afilt1c		butterbp anoise, kcps1*3, kcps1*0.05
	afilt1d		butterbp anoise, kcps1*4, kcps1*0.05
	afilt1e		butterbp anoise, kcps1*5, kcps1*0.05
	asum		sum afilt1a, afilt1b, afilt1c, afilt1d, afilt1e
	aout		butterbp asum*5+(anoise*0.01), kcentro1, kcentro1*0.2
	aout		= aout*krms1*10
			chnmix aout, "MasterOut1"
			chnmix aout, "MasterOut2"
/*
	acps		upsamp kcps1/2000
	acentro		upsamp kcentro1/2000
	arms		upsamp krms1
			fout "testplay.wav", 14, aout, acps, acentro, arms, aenv
*/
	endin

; ******************************
; partikkel instr 
	instr 52			
;krms1 		chnget "respondLevel1"
;kactive = (krms1 > 0 ? 1 : 0)
;printk2 kactive  

#include "partikkel_chn.inc"
#include "partikkel_self.inc"
			chnmix a1, "MasterOut1"
			chnmix a2, "MasterOut2"
	endin

; ******************************
; spectral resynthesis instr 
	instr 53			
	kinit		init 1
	if kinit > 0 then
	a1		= 0
	fsin 		pvsanal	a1, gifftsize, gifftsize/4, gifftsize, 1
	endif
	;fsin		pvsinit gifftsize, gifftsize/4, gifftsize, 1
			pvsftr	fsin,gifnaResyn,gifnfResyn		;read modified data back to fsrc
	aout		pvsynth	fsin				;and resynth
			chnmix aout, "MasterOut1"
			chnmix aout, "MasterOut2"
	endin

	instr 80
; test table
	ksum tabsum gifnaResyn
	printk2 ksum
	endin


; ******************************
; self analysis of own output 
	instr 98
	a1	chnget "MasterOut1"
	a2	= 0
	ifna	= gifnaSelf
	ifnf	= gifnfSelf
#include "audio_analyze.inc"
			chnset kflag, "pvsoutflag"
; write to chn
			chnset kstatus, "myAudioStatus"
			chnset kstatusTrig, "myAudioStatusTrig"
			chnset ktrig1, "myTransient"
			chnset krms1, "myLevel1"
			chnset kcps1, "myPitch1ptrack"
			chnset kcps1p, "myPitch1pll"
			chnset kautocorr, "myAutocorr1"
			chnset kcentroid, "myCentroid1"
			chnset kspread, "mySpread1"
			chnset kskewness, "mySkewness1"
			chnset kurtosisM, "myKurtosis1"
			chnset kflatness, "myFlatness1"
			chnset kcrest, "myCrest1"
			chnset kflux, "myFlux1"
			chnset kepochSig, "myEpochSig1"
			chnset kepochRms, "myEpochRms1"
			chnset kepochZCcps, "myEpochZCcps1"

	endin

; ******************************
; master output 
	instr 99

	a1	chnget "MasterOut1"
	a2	chnget "MasterOut2"
	a0	= 0
		outs a1, a2
		chnset (a1+a2)*0.5, "MyOutput"
		chnset a0, "MasterOut1"
		chnset a0, "MasterOut2"

	endin
</CsInstruments>

<CsScore>
; run for N sec
;i3 	0 86400				; audio file input
i4 	0 86400				; audio input
i11 	0 86400				; merge left and right input
i21 	0 .1 1				; initialize input level
i22 	0 .1 "inputNoisefloor" -30	; initialize noise floor
i31 	0 86400				; analysis
;i51 	0 -1				; subtractive harmonic resynthesis
i52 	0 -1				; partikkel resynthesis
;i53 	3 -1				; fft resynthesis
;i98 	0 86400				; analysis of own output
;i80 	0 86400				; test print for contents of fft tables
i99 	0 86400				; master out

</CsScore>

</CsoundSynthesizer>
