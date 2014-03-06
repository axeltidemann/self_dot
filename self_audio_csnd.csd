<CsoundSynthesizer>
<CsOptions>
-odac
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 32
	nchnls = 2	
	0dbfs = 1

	giSine		ftgen	0, 0, 65536, 10, 1

;**************
	instr 1  
	iamp	= ampdbfs(-5)
	icps	= 220
	koffset	chnget "freq_offset"
  	kcps	= icps + koffset

	a1 	oscili iamp, kcps, giSine	; sine test tone
  	a2 	oscili iamp, kcps*2, giSine	; sine test tone 2
	;a1,a2	inch 1,2			; for audio in to replace the sine test tones



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

; ***************
; dump fft to python
         	;pvsout  fftin1, 0 			; write signal to pvs out bus channel 0
         	;pvsout  fftin1, 1			; write signal to pvs out bus channel 0

; ***************
; pitch tracking

	kcps1 		init 0	
	kcps2 		init 0	
	ihopsize	= 512
	kcps1a = 200;, kamp1 	ptrack a1, ihopsize
	kcps2a = 200;, kamp2 	ptrack a2, ihopsize

	kcps1b		init 100
	kcps1b		= (kcps1a > 700 ? kcps1b : kcps1a) ; avoid high intermittencies
	kcps1		init 100
	kcps1		= (krms1 < 0.04 ? kcps1 : kcps1b)  ; don't track (keep last value) at low amplitude

	kcps2b		init 100
	kcps2b		= (kcps2a > 700 ? kcps2b : kcps2a) 
	kcps2		init 100
	kcps2		= (krms2 < 0.04 ? kcps2 : kcps2b)

; filter pitch tracking signal
	imedianSize	= ihopsize
	kcps1		mediank	kcps1, imedianSize, imedianSize
	kcps2		mediank	kcps2, imedianSize, imedianSize


; ***************
; write to chn
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

</CsInstruments>

<CsScore>
; run for N sec
i1 0 10

</CsScore>

</CsoundSynthesizer>
