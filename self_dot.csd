<CsoundSynthesizer>
<CsOptions>
;-odac1 -iadc0 
;-odac18 -iadc20
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 32
	nchnls = 2	
	0dbfs = 1

	giSine		ftgen	0, 0, 65536, 10, 1

; pvs ftables
	gifftsize 	= 256
	giFftTabSize	= (gifftsize / 2)+1
	gifna     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis
	gifnf     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   ; make ftable for pvs analysis

;******************************
; audio file input 
	instr 3

/*
	; test tone
	iamp	= ampdbfs(-5)
	icps	= 220
	koffset	chnget "freq_offset"
  	kcps	= icps + koffset
	a1 	oscili iamp, kcps, giSine	; sine test tone
  	a2 	oscili iamp, kcps*2, giSine	; sine test tone 2
*/

	a1	soundin "bkup/fox.wav"
	a2	= 0
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
; ***************
; extract stereo position here, then collapse input signal to mono
	kpos		= 0.5	; ...as if we already had implemented position tracking
	a1		= a1+a2

; ***************
; amplitude tracking
	krms1		rms a1				; simple level measure
	kAttack		= 0.01
	kRelease	= 0.3
	aFollow1	follow2	a1, kAttack, kRelease	; envelope follower
	kFollow1	downsamp aFollow1	

; ***************
; epoch filtering
	a20		butterbp a1, 20, 5
	a20		dcblock2 a20*40
	aepochSig	butlp a20, 200
	kepochSig	downsamp aepochSig
	kepochRms	rms aepochSig

; count epoch zero crossings
	ktime		times	
	kZC		trigger kepochSig, 0, 0
	kprevZCtim	init 0
	kinterval1	init 0
	kinterval2	init 0
	kinterval3	init 0
	kinterval4	init 0
	if kZC > 0 then
	kZCtim	 	= ktime
	kinterval4	= kinterval3
	kinterval3	= kinterval2
	kinterval2	= kinterval1
	kinterval1	= kZCtim-kprevZCtim
	kprevZCtim	= kZCtim
	endif
	kmax		max kinterval1, kinterval2, kinterval3, kinterval4
	kmin		min kinterval1, kinterval2, kinterval3, kinterval4
	kZCmedi		= (kinterval1+kinterval2+kinterval3+kinterval4-kmax-kmin)/2
	kepochZCcps	divz 1, kZCmedi, 1


; ***************
; spectral analysis


	iwtype 			= 1
	fsin 			pvsanal	a1, gifftsize, gifftsize/4, gifftsize, iwtype
	kflag   		pvsftw	fsin,gifna,gifnf          	; export  amps  and freqs to table,

	kupdateRate		= 1000	; correlation interval
	kflatness		init -1
	kmetro			metro kupdateRate
	kdoflag			init 0
	kdoflag			= kdoflag + kmetro

	; copy pvs data from table to array
	; analyze spectral features
	kArrA[]  		init    giFftTabSize-2
	kArrAprev[]  		init    giFftTabSize-2
	kArrF[]  		init    giFftTabSize-2
	kArrCorr[]  		init    giFftTabSize-2
	kflatness		init -1

if (kdoflag > 0) && (kflag > 0) then
	kArrAprev[]		= kArrA
        			copyf2array kArrA, gifna
        			copyf2array kArrF, gifnf	
	kindx 			= 0
	kcentroid		= 0
	ksumAmp			sumarray kArrA
	kflatsum		= 0
	kflatlogsum		= 0
	kcorrSum		= 0
	kthisSum2		= 0
	kprevSum2		= 0

  process:
	kArrCorr[kindx]		= kArrA[kindx]*kArrAprev[kindx]
	knormAmp		= kArrA[kindx] / ksumAmp
	kcentroid		= kcentroid + (kArrF[kindx]*knormAmp)
	kflatsum		= kflatsum + kArrA[kindx]
	kflatlogsum		= kflatlogsum + log(kArrA[kindx])
	kcorrSum		= kcorrSum + (kArrAprev[kindx]*kArrA[kindx])
	kprevSum2		= kprevSum2 + (kArrAprev[kindx]^2)
	kthisSum2		= kthisSum2 + (kArrA[kindx]^2)
	kindx 			= kindx + 1
  if kindx < giFftTabSize-2 then
  kgoto process
  endif

; separate loop for spread
	kindx 			= 0
	kspread			= 0
	kskewness		= 0
	kurtosis		= 0
  spread:
	knormAmp		= kArrA[kindx] / ksumAmp
	kspread			= ((kArrF[kindx] - kcentroid)^2)*knormAmp
	kskewness		= ((kArrF[kindx] - kcentroid)^3)*knormAmp
	kurtosis		= ((kArrF[kindx] - kcentroid)^4)*knormAmp
	kindx 			= kindx + 1
  if kindx < giFftTabSize-2 then
  kgoto spread
  endif
	kspread			= kspread^0.5
	kskewness		= (kskewness / (kspread^3))/200000
	kurtosis		= kurtosis / (kspread^4)/1000000000
	kflatness		= exp(1/(giFftTabSize-2)*kflatlogsum) / (1/(giFftTabSize-2)*kflatsum)
	kmaxAmp			maxarray kArrA
	kcrest			= kmaxAmp / (1/(giFftTabSize-2)*kflatsum)
	kflux			= 1-(kcorrSum / (sqrt(kprevSum2)*sqrt(kthisSum2)))
	kdoflag 		= 0
endif

	kautocorr		sumarray kArrCorr
	krmsA			sumarray kArrA
	krmsAprev		sumarray kArrAprev
	kautocorr		= (kautocorr / (krmsA*krmsAprev)*1.79)

; ***************
; dump fft to python
         	;pvsout  fftin1, 0 			; write signal to pvs out bus channel 0

; ***************
; pitch tracking
; using two different methods, keeping both signals
; ptrack may be better for polyphonic signals
; plltrack probably better for speech


	kcps1 		init 0	
	ihopsize	= 512

	kcps1a, kamp1 	ptrack a1, ihopsize

	kcps1b		init 100
	kcps1b		= (kcps1a > 800 ? kcps1b : kcps1a) ; avoid high intermittencies
	kcps1		init 100
	kcps1		= (krms1 < 0.03 ? kcps1 : kcps1b)  ; don't track (keep last value) at low amplitude

	imedianSize	= ihopsize
	kcps1		mediank	kcps1, imedianSize, imedianSize

	acps1p, alock1p	plltrack a1, 0.2 
	kcps1p		downsamp acps1p
	imedianSize	= 200
	kcps1p		mediank	kcps1p, imedianSize, imedianSize

; ***************
; write to chn
			chnset krms1, "level1"
			chnset kFollow1, "envelope1"
			chnset kcps1, "pitch1ptrack"
			chnset kcps1p, "pitch1pll"
			chnset kautocorr, "autocorr1"
			chnset kcentroid, "centroid1"
			chnset kspread, "spread1"
			chnset kskewness, "skewness1"
			chnset kurtosis, "kurtosis1"
			chnset kflatness, "flatness1"
			chnset kcrest, "crest1"
			chnset kflux, "flux1"
			chnset kepochSig, "epochSig1"
			chnset kepochRms, "epochRms1"
			chnset kepochZCcps, "epochZCcps1"


;  		out a1,a2
	endin


; ******************************
; resynthesis/imitation
; only using channel 1 for now

	instr 9


	krms1 		chnget "respondLevel1"
	kenv1 		chnget "respondEnvelope1"
	kcps1 		chnget "respondPitch1ptrack"
	kcps1p 		chnget "respondPitch1pll"
	kcentro1 	chnget "respondCentroid1"

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
	aout		= aout*kenv1*1
			outs aout, aout
	endin

</CsInstruments>

<CsScore>
; run for N sec
;i3 0 86400	; audio file input
i4 0 86400	; audio input
i5 0 86400	; analysis
i9 0 86400	; resynthesis

</CsScore>

</CsoundSynthesizer>
