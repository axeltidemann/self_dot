<CsoundSynthesizer>
<CsOptions>
;-odac1 -iadc0 
-odac18 -iadc20
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 64
	nchnls = 2	
	0dbfs = 1

; pvs ftables
	gifftsize 	= 256
	giFftTabSize	= (gifftsize / 2)+1
	gifna     	ftgen   1 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis
	gifnf     	ftgen   2 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis
	gifnaSelf     	ftgen   3 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis of my own output
	gifnfSelf     	ftgen   4 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis of my own output
	gifnaResyn     	ftgen   11 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs resynthesis
	gifnfResyn     	ftgen   12 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs resynthesis

; classic waveforms
	giSine		ftgen	0, 0, 65536, 10, 1					; sine wave
	giCosine	ftgen	0, 0, 8192, 9, 1, 1, 90					; cosine wave
	giTri		ftgen	0, 0, 8192, 7, 0, 2048, 1, 4096, -1, 2048, 0		; triangle wave 

; grain envelope tables
	giSigmoRise 	ftgen	0, 0, 8193, 19, 0.5, 1, 270, 1				; rising sigmoid
	giSigmoFall 	ftgen	0, 0, 8193, 19, 0.5, 1, 90, 1				; falling sigmoid
	giExpFall	ftgen	0, 0, 8193, 5, 1, 8193, 0.00001				; exponential decay
	giTriangleWin 	ftgen	0, 0, 8193, 7, 0, 4096, 1, 4096, 0			; triangular window 

;***************************************************
;user defined opcode, asynchronous clock
;***************************************************
			opcode		probabilityClock, a, k
	kdens		xin
			setksmps 1
	krand		rnd31	1, 1
	krand		= (krand*0.5)+0.5
	ktrig		= (krand < kdens/kr ? 1 : 0)
	atrig		upsamp ktrig
			xout atrig
			endop

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
; analysis  of audio input 
	instr 5
	
	a1		chnget "in1"
	a2		chnget "in2"
	a0		= 0
			chnset a0, "in1"
			chnset a0, "in2"

#include "audio_analyze.inc"

; ***************
; write to chn
	kcentroidG	= kcentroid*kgate	; limit noise contribution in quiet sections
	kautocorrG	= kautocorr * kgate	
	kspreadG	= kspread * kgate
	kskewnessG	= kskewness * kgate
	kurtosisM	mediank kurtosis, 6, 6
	kflatnessG	= kflatness * kgate
	kfluxG		= kflux * kgate

			chnset kstatus, "audioStatus"
			chnset krms1, "level1"
			chnset kFollow1, "envelope1"
			chnset kcps1, "pitch1ptrack"
			chnset kcps1p, "pitch1pll"
			chnset kautocorrG, "autocorr1"
			chnset kcentroidG, "centroid1"
			chnset kspreadG, "spread1"
			chnset kskewnessG, "skewness1"
			chnset kurtosisM, "kurtosis1"
			chnset kflatnessG, "flatness1"
			chnset kcrest, "crest1"
			chnset kfluxG, "flux1"
			chnset kepochSig, "epochSig1"
			chnset kepochRms, "epochRms1"
			chnset kepochZCcps, "epochZCcps1"

  		out a1,a2
	endin


; ************************************************************
; resynthesis/imitation

; ******************************
; basic subtractive harmonic synth
	instr 11

	krms1 		chnget "respondLevel1"
	kenv1 		chnget "respondEnvelope1"
	kcps1 		chnget "respondPitch1ptrack"
	kcps1p 		chnget "respondPitch1pll"
	kcentro1 	chnget "respondCentroid1"

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
	aout		= aout*kenv1*1
			chnset aout, "MasterOut1"
			chnset aout, "MasterOut2"
	endin

; ******************************
; partikkel instr 
	instr 12
#include "partikkel_chn.inc"
#include "partikkel_self.inc"
			chnset a1, "MasterOut1"
			chnset a2, "MasterOut2"
	endin

; ******************************
; spectral resynthesis instr 
	instr 13
	fsrc		pvsinit gifftsize, gifftsize/4, gifftsize, 1
			pvsftr	fsrc,gifnaResyn,gifnfResyn		;read modified data back to fsrc
	aout		pvsynth	fsrc				;and resynth
			chnset aout, "MasterOut1"
			chnset aout, "MasterOut2"
	endin

; ******************************
; self analysis of own output 
	instr 98
	a1	chnget "MasterOut1"
	a2	= 0
#include "audio_analyze.inc"
; write to chn
	kcentroidG	= kcentroid*kgate	; limit noise contribution in quiet sections
	kautocorrG	= kautocorr * kgate	
	kspreadG	= kspread * kgate
	kskewnessG	= kskewness * kgate
	kurtosisM	mediank kurtosis, 6, 6
	kflatnessG	= kflatness * kgate
	kfluxG		= kflux * kgate

			chnset kstatus, "myAudioStatus1"
			chnset krms1, "myLevel1"
			chnset kFollow1, "myEnvelope1"
			chnset kcps1, "myPitch1ptrack"
			chnset kcps1p, "myPitch1pll"
			chnset kautocorrG, "myAutocorr1"
			chnset kcentroidG, "myCentroid1"
			chnset kspreadG, "mySpread1"
			chnset kskewnessG, "mySkewness1"
			chnset kurtosisM, "myKurtosis1"
			chnset kflatnessG, "myFlatness1"
			chnset kcrest, "myCrest1"
			chnset kfluxG, "myFlux1"
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
		chnset a0, "MasterOut1"
		chnset a0, "MasterOut2"

	endin
</CsInstruments>

<CsScore>
; run for N sec
;i3 0 86400	; audio file input
i4 0 86400	; audio input
i5 0 86400	; analysis
i11 0 86400	; subtractive harmonic resynthesis
;i12 0 86400	; partikkel resynthesis
;i13 0 86400	; fft resynthesis
;i98 0 86400	; analysis of own output
i99 0 86400	; master out

</CsScore>

</CsoundSynthesizer>
