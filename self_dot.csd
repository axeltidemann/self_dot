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
	krms		rms a1
	kgate		= (krms < ampdbfs(-30) ? 0 : 1)	; don't bother analyzing very quiet sections
	kgate		tonek kgate, 10
	a1		= a1 * kgate

; ***************
; extract stereo position here, then collapse input signal to mono
	kpos		= 0.5	; ...as if we already had implemented position tracking
	a1		= a1+a2

; ***************
; amplitude tracking
	krms1		rms a1				; simple level measure
	kAttack		= 0.001				; envelope follower attack
	kRelease	= 0.3 				; envelope follower release
	aFollow1	follow2	a1, kAttack, kRelease	; envelope follower
	kFollow1	downsamp aFollow1	

; ***************
;analyze transients
	iResponse	= 10 		; response time in milliseconds
	kAtck_db	= 3		; attack threshold (in dB)
	iLowThresh	= 0.005		; lower threshold for transient detection (adaptive, relative to recent transient strength)
	idoubleLimit	= 0.02		; minimum duration between events, (double trig limit)
	
	kFollowdb1	= dbfsamp(kFollow1)			; convert to dB
	kFollowDel1	delayk	kFollowdb1, iResponse/1000	; delay with response time for comparision of levels
	kTrig1		init 0
	kLowThresh1	init 0
	kTrig1		= ((kFollowdb1 > kFollowDel1 + kAtck_db) ? 1 : 0) 	; if current rms plus threshold is larger than previous rms, set trig signal to current rms
	
	; avoid transient detection of very soft signals (adaptive level)
	if kTrig1 > 0 then
	kLowThresh1	= (kLowThresh1*0.7)+(kFollow1*0.3)		; (the coefficients can be used to adjust adapt rate)
	endif
	kTrig1		= (kLowThresh1 > iLowThresh ? kTrig1 : 0)
	
	; avoid closely spaced transient triggers (first trig priority)
	kDouble1	init 1
	kTrig1		= kTrig1*kDouble1
	if kTrig1 > 0 then
	reinit double1
	endif
	double1:
	kDouble1	linseg	0, idoubleLimit, 0, 0, 1, 1, 1
	rireturn
		
; ***************
; segmentation, set kstatus = 1 when a transient occurs, 
; keep status = 1 until level drops 6dB below the level at which the transient was detected
; and keep status = 1 for a release time period after the level has dropped below the threshold
	iStatusThresh	= 6			; status release threshold, signal must drop this much below transient level before we say that the audio segment is finished
	iStatusRel	= 0.01			; status release time, hold status=1 for this long after signal has dropped below threshold
	kstate		init 0
	kstate		= (kTrig1 > 0 ? 1 : kstate)
	ksegStart	trigger kstate, 0.5, 0
	kdBStart	init 0
	kdBStart 	= (ksegStart > 0 ? kFollowdb1 : kdBStart)
	kstate		= (kFollowdb1 < (kdBStart-iStatusThresh) ? 0 : kstate)
	kstate_Dly	delayk kstate, iStatusRel
	kstate_Rel	= ((kstate == 0) && (kstate_Dly == 0)? 0 : 1)
	kstatus		= kstate_Rel

; ***************
; epoch filtering
	a20		butterbp a1, 20, 5
	a20		dcblock2 a20*40
	aepochSig	butlp a20, 200
	kepochSig	downsamp aepochSig
	kepochRms	rms aepochSig

; count epoch zero crossings
	ktime		times	
	kZC		trigger kepochSig, 0, 0		; zero cross
	kprevZCtim	init 0
	kinterval1	init 0
	kinterval2	init 0
	kinterval3	init 0
	kinterval4	init 0
	if kZC > 0 then
	kZCtim	 	= ktime				; get time between zero crossings
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

	kupdateRate		= 1000	
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
	knormAmp		divz kArrA[kindx], ksumAmp, 0
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

; separate loop for spread, skewness, kurtosis (as they depend on centroid being previously calculated) 
	kindx 			= 0
	kspread			= 0
	kskewness		= 0
	kurtosis		= 0
  spread:
	knormAmp		divz kArrA[kindx], ksumAmp, 0
	kspread			= ((kArrF[kindx] - kcentroid)^2)*knormAmp
	kskewness		= ((kArrF[kindx] - kcentroid)^3)*knormAmp
	kurtosis		= ((kArrF[kindx] - kcentroid)^4)*knormAmp
	kindx 			= kindx + 1
  if kindx < giFftTabSize-2 then
  kgoto spread
  endif
	kflat_1			divz 1, (giFftTabSize-2)*kflatlogsum, 1
	kflat_2			divz 1, (giFftTabSize-2)*kflatsum, 1
	kflatness		= exp(kflat_1) / kflat_2
	kspread			= kspread^0.5
	kskewness		divz kskewness, kspread^3, 0
	kurtosis		divz kurtosis, (kspread^4), 0
	kmaxAmp			maxarray kArrA
	kcrest			= kmaxAmp / kflat_2
	kflux_1			divz kcorrSum, (sqrt(kprevSum2)*sqrt(kthisSum2)), 1
	kflux			= 1-kflux_1
	kdoflag 		= 0
endif
	kautocorr		sumarray kArrCorr
	krmsA			sumarray kArrA
	krmsAprev		sumarray kArrAprev
	kautocorr		divz kautocorr*2, (krmsA*krmsAprev) , 0

; ***************
; dump fft to python
         			pvsout  fsin, 0 			; write signal to pvs out bus channel 0

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
