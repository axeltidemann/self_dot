<CsoundSynthesizer>
<CsOptions>
;-odac18 -iadc20
;-odac34 -m256 -B512
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 32
	nchnls = 2	
	0dbfs = 1

	giSine		ftgen	0, 0, 65536, 10, 1

;******************************
; analysis  of audio input 
	instr 1  

/*
	; test tone
	iamp	= ampdbfs(-5)
	icps	= 220
	koffset	chnget "freq_offset"
  	kcps	= icps + koffset
	a1 	oscili iamp, kcps, giSine	; sine test tone
  	a2 	oscili iamp, kcps*2, giSine	; sine test tone 2
*/
/*
	kmetro	metro 0.2
	if kmetro > 0 then
	reinit play
	endif
play:
	irnd	rnd31 0.3, 1
	ispeed	= 1 + irnd
	a1	diskin "fox.wav", ispeed, 0
rireturn
*/
;	a1	diskin "TheColourofMagic_pauses.wav", 1, 1, 1
;	a2	= 0

	; live audio input
	a1,a2	inch 1,2

; ***************
; segmentation, set kstatus = 1 when something interesting happens (spoken sentence or some other coherent sound)
	kattack		= 0.01
	krelease	= 2.0
	arms		follow2 a1, kattack, krelease
	krms		downsamp arms*1.6
	krms_dB		= dbfsamp(krms)
	iAttackThresh	= -25
	iReleaseThresh	= iAttackThresh-6
	kstate 		init  0 	
	if krms_dB > iAttackThresh then
	kstate		= 1
	endif
	if (kstate == 1) && (krms_dB > iReleaseThresh) then
	kstate		= 1
	else
	kstate 		= 0
	endif
	; fade in and out, cleanly separate segments from noise floor
	astate		upsamp kstate
	astate		butterlp astate, 5
	a1del		delay a1, 0.1
	a1		= a1del * astate



; ***************
; amplitude tracking
	krms1		rms a1				; simple level measure
	krms2		rms a2
	kAttack		= 0.01
	kRelease	= 0.3
	aFollow1	follow2	a1, kAttack, kRelease	; envelope follower
	kFollow1	downsamp aFollow1	
	aFollow2	follow2	a2, kAttack, kRelease
	kFollow2	downsamp aFollow2	

; ***************
; centroid
	irefrtm		= 20				; interval for generating new values for the spectral centroid
	ifftsize	= 1024
	ioverlap	= ifftsize / 4
	iwinsize	= ifftsize
	iwinshape	= 1				; von-Hann window
	fftin1		pvsanal	a1, ifftsize, ioverlap, iwinsize, iwinshape
	fftin2		pvsanal	a2, ifftsize, ioverlap, iwinsize, iwinshape
	ktrig		metro irefrtm
if ktrig == 1 then
	kcentro1	pvscent	fftin1
	kcentro2	pvscent	fftin2
endif
	kcentro1	tonek kcentro1, 100	
	kcentro2	tonek kcentro2, 100	

; ***************
; dump fft to python
         	;pvsout  fftin1, 0 			; write signal to pvs out bus channel 0
         	;pvsout  fftin1, 1			; write signal to pvs out bus channel 0

; ***************
; pitch tracking
; choose from two different methods 
; ptrack may be better for polyphonic signals
; plltrack probably better for speech

/*
	kcps1 		init 0	
	kcps2 		init 0	
	ihopsize	= 512

	kcps1a, kamp1 	ptrack a1, ihopsize
	kcps2a, kamp2 	ptrack a2, ihopsize

	kcps1b		init 100
	kcps1b		= (kcps1a > 800 ? kcps1b : kcps1a) ; avoid high intermittencies
	kcps1		init 100
	kcps1		= (krms1 < 0.03 ? kcps1 : kcps1b)  ; don't track (keep last value) at low amplitude

	kcps2b		init 100
	kcps2b		= (kcps2a > 800 ? kcps2b : kcps2a) 
	kcps2		init 100
	kcps2		= (krms2 < 0.03 ? kcps2 : kcps2b)

	imedianSize	= ihopsize
	kcps1		mediank	kcps1, imedianSize, imedianSize
	kcps2		mediank	kcps2, imedianSize, imedianSize
*/

	acps1, alock1 	plltrack a1, 0.2 
	kcps1		downsamp acps1
	acps2, alock2 	plltrack a2, 0.2 
	kcps2		downsamp acps2
	imedianSize	= 200
	kcps1		mediank	kcps1, imedianSize, imedianSize
	kcps2		mediank	kcps2, imedianSize, imedianSize



; ***************
; write to chn
			chnset kstate, "audioStatus"
			chnset krms1, "level1"
			chnset krms2, "level2"
			chnset kFollow1, "envelope1"
			chnset kFollow2, "envelope2"
			chnset kcps1, "pitch1"
			chnset kcps2, "pitch2"
			chnset kcentro1, "centroid1"
			chnset kcentro2, "centroid2"

  		out a1,a2
	endin


; ******************************
; resynthesis/imitation
; only using channel 1 for now

	instr 2


	krms1 		chnget "imitateLevel1"
	kenv1 		chnget "imitateEnvelope1"
	kcps1 		chnget "imitatePitch1"
	kcentro1 	chnget "imitateCentroid1"
	krms1		limit krms1, 0, 1
	kenv1		limit kenv1, 0, 1
	kcps1		limit kcps1, 20, 10000
	kcentro1	limit kcentro1, 20, 10000

/*
	imedianSize	= 200
	krms1		mediank krms1, imedianSize, imedianSize
	kenv1 		mediank kenv1, imedianSize, imedianSize
	kcps1 		mediank kcps1, imedianSize, imedianSize
	kcentro1	mediank kcentro1, imedianSize, imedianSize
*/

/*
	; only for csound standalone testing
	krms1 		chnget "level1"
	kenv1 		chnget "envelope1"
	kcps1 		chnget "pitch1"
	kcentro1 	chnget "centroid1"
*/

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
	a0		= 0
			outs a0, aout
	endin

</CsInstruments>

<CsScore>
; run for N sec
i1 0 100
i2 0 100

</CsScore>

</CsoundSynthesizer>
