
; ***************
; pre-emphasis EQ for transient detection,
; allowing better sensitivity to utterances starting with a sibliant.
	ktd_HiShelfFq	chnget "td_HiShelfFq"
	ktd_HiShelfGain	chnget "td_HiShelfGain"
	ktd_LoShelfFq	chnget "td_LoShelfFq"
	ktd_LoShelfGain	chnget "td_LoShelfGain"
	ktd_AmpAdjust	chnget "td_AmpAdjust"
	
	atd_input	pareq a1, ktd_HiShelfFq, ampdb(ktd_HiShelfGain), 0.7,  2
        atd_input	pareq atd_input, ktd_HiShelfFq, ampdb(ktd_LoShelfGain), 0.7,  1
	atd_input	= atd_input*ampdb(ktd_AmpAdjust)


; ***************
; amplitude tracking
	krms1		rms atd_input			; simple level measure
	kAttack		= 0.001				; envelope follower attack
	kRelease	= 0.3 				; envelope follower release
	aFollow1	follow2	atd_input, kAttack, kRelease	; envelope follower
	kFollow1	downsamp aFollow1	
	kFollowdb1	= dbfsamp(kFollow1)		; convert to dB

	knoiseFloor_dB	chnget "inputNoisefloor"
	kgate		= (dbfsamp(krms1) < knoiseFloor_dB ? 0 : 1)	; don't bother analyzing very quiet sections
	kgatesoft	tonek kgate, 4
	krms1gated	= krms1*kgatesoft

; ***************
;analyze transients
	iresponse	= 10 			; response time in milliseconds
	ktthresh	= 9			; transient trig threshold 
	klowThresh	= knoiseFloor_dB	; lower threshold for transient detection
	idoubleLimit	= 0.02			; minimum duration between events, (double trig limit)
	kdecThresh	= 6			; retrig threshold, how much must the level decay from its local max before allowing new transient trig

	ktrig1, kdiff	TransientDetect kFollowdb1, iresponse, ktthresh, klowThresh, kdecThresh, idoubleLimit

; ***************
; segmentation, set kstatus = 1 when a transient occurs, 
; keep status = 1 until level drops 6dB below the level at which the transient was detected
; and keep status = 1 for a release time period after the level has dropped below the threshold
	iStatusThresh	= 9					; status release threshold, signal must drop this much below transient level before we say that the audio segment is finished
	iStatusRel	= 0.8;0.2					; status release time, hold status=1 for this long after signal has dropped below threshold
;	iDecResponse	= 100					; decay rate analysis response time
;	krms1F		tonek krms1, 5
;	krms1F_dB	= ampdbfs(krms1F)
;	krms1F_dBdel	delayk	krms1F_dB, iDecResponse/1000	; delay with response time for comparision of levels
;	kDecayRate_dB	limit krms1F_dBdel-krms1F_dB, 0, 1		; decay rate  ...
;	kStatusRel2	= (kDecayRate_dB < .01 ? (.01-kDecayRate_dB)*5 : 1/kr) ; prolong status release when decay rate is very slow
	kstate		init 0
	kstate		= (ktrig1 > 0 ? 1 : kstate)
	ksegStart	trigger kstate, 0.5, 0
	kdBStart	init 0
	kdBStart 	= (ksegStart > 0 ? kFollowdb1 : kdBStart)
	kstate		= (kFollowdb1 < (kdBStart-iStatusThresh) ? 0 : kstate)
	kstaTrigRel	trigger kstate, 0.5, 1		; up-down trig
	if kstaTrigRel > 0 then
	reinit releaseState
	endif
releaseState:
	kreleaseFlag	linseg 1, iStatusRel, 0, .1, 0
	kreleaseFlagT	trigger kreleaseFlag, 0.02, 1		; trig when release ends
rireturn
;	if kreleaseFlagT > 0 then 
;	reinit release2
;	endif
;release2:
;	kreleaseFlag2	linseg 1, i(kStatusRel2), 0, .1, 0
;rireturn
	kstate_Rel	= ((kstate == 0) && (kreleaseFlag < 0.01)? 0 : 1)
	kstatus		= kstate_Rel
	kstatusTrigOn	trigger kstatus, 0.5, 0
	kstatusTrigOff	trigger kstatus, 0.5, 1
	kstatusTrig	= kstatusTrigOn-kstatusTrigOff	; pulse 1 at start, pulse -1 at end of segment
/*

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
	kflag   		pvsftw	fsin,ifna,ifnf          	; export  amps  and freqs to table,

	kupdateRate		= 1000	
	kflatness		init -1
	kmetro			metro kupdateRate
	kdoflag			init 0
	kdoflag			= (kdoflag + kmetro);*kgate

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
	kcentroid		samphold kcentroid, kgate

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
*/
; ***************
/*
;analyze centroid transients
	kcentroid	limit kcentroid, 20, sr/2
	kcentrolog	= log2(kcentroid)
	acentro		upsamp kcentrolog
	ac		upsamp kcentroid
	kcAttack	= 0.001
	kcRelease	= 0.1
	aFolCentro	follow2	acentro, kcAttack, kcRelease	; envelope follower
	kFolCentro	downsamp aFolCentro	

	icresponse	= 10 		; response time in milliseconds
	kctthresh	= 0.5		; transient trig threshold (in octaves)
	kclowThresh	= 12		; lower threshold for transient detection (in octaves)
	icdoubleLimit	= 0.02		; minimum duration between events, (double trig limit)
	kcdecThresh	= .5		; retrig threshold, how much must the level decay from its local max before allowing new transient trig

	kctrig1, kcdiff	TransientDetect kFolCentro, icresponse, kctthresh, kclowThresh, kcdecThresh, icdoubleLimit
*/

/*
; ***************
; dump fft to python
         	;pvsout  fftin1, 0 			; write signal to pvs out bus channel 0

; ***************
; pitch tracking
; using two different methods, keeping both signals
; ptrack may be better for polyphonic signals
; plltrack probably better for speech


	kcps1 		init 0	
	ihopsize	= 1024

	kcps1, kamp1 	ptrack a1, ihopsize

	kcps1		samphold kcps1, kgate
	kcpsRangeGate	= (kcps1 > 600? 0 : 1)
	kcps1		samphold kcps1, kcpsRangeGate
	kpgate		= (kcentroid > 7000 ? 0 : 1)
	kcps1		samphold kcps1, kpgate
	imedianSize	= 512
	kcps1		mediank	kcps1, imedianSize, imedianSize

	kd 		= 0.27
	kloopf		= 40
	kloopq		= 0.3
	klf		= 50
	khf		= 1500
	kthresh		= ampdbfs(knoiseFloor_dB-8)
	acps1p, alock1p	plltrack a1, kd, kloopf, kloopq, klf, khf, kthresh 
	kcps1p		downsamp acps1p
	;kcps1p		samphold kcps1p, kgate
	imedianSize	= 20
	kcps1p		mediank	kcps1p, imedianSize, imedianSize

; ***************
; post filtering
	kautocorr	samphold kautocorr, kgate	
	kspread		samphold kspread, kgate
	kurtosisM	mediank kurtosis, 6, 6

	kcpsBoth	= kepochZCcps+kcps1p
	kpzgate		= (kcentroid > 6000 ? 0 : 1)
	kepochZCcps	samphold kepochZCcps, kpzgate
*/

