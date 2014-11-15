;******************************
; analysis  of audio input 
	instr 31
	
	a1		chnget "in1"
	kinlevel	chnget "inputLevel"
	a1		= a1*kinlevel
	ifna		= gifna
	ifnf		= gifnf
#include "audio_analyze.inc"

; ***************
; write to chn

	Stest		sprintfk "input audio status %i, trig status %i, dbstart %f", kstatus, kstatusTrig, kdBStart
			;puts Stest, 2+(kstatus + kstatusTrig)

			chnset kstatus, "audioStatus"
			chnset kstatusTrig, "audioStatusTrig"
			chnset ktrig1, "transient"
            chnset kcpsA, "pitch1_amdf"
			chnset krms0, "in_amp"
			chnset kFollowdb1, "followdb"
			chnset kcps1, "in_pitch"
			chnset kcentroid, "in_centroid"
			
/*
			chnset kflag, "pvsinflag"
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


        ; for internal use
			chnset krms1gated, "level1gated"
*/
        ; send audio to memory recording
	arec            delay a1, 0.02  ; delay to sync with transient detection (and then add some to capture a bit of potential soft starts like 'h')
                        chnset arec, "rec_in1"
                        
        ; enable by setting kinputMonitor if we want to monitor the input signal
	kinputMonitor	chnget "inputMonitor"
			chnmix a1*kinputMonitor, "MasterOut1"
			chnmix a1*kinputMonitor, "MasterOut2"
/*
; *** test write analysis output
iwritetest	= 0
if iwritetest > 0 then
	astatus		upsamp	kstatus
	astatusTrig	upsamp	kstatusTrig
	atrig1		upsamp	ktrig1
	arms1gated	upsamp	krms1gated*2.5
	arms1		upsamp	krms1*2.5
	acps1		upsamp	kcps1/500
	acps1p		upsamp	kcps1p/500
	aautocorr	upsamp	kautocorr
	acentroid	upsamp	kcentroid/15000
	aspread		upsamp	kspread/1200
	askewness	upsamp	kskewness/2000
	aurtosisM	upsamp	kurtosisM/1000000
	aflatness	upsamp	kflatness/500
	acrest		upsamp	kcrest/100
	aflux		upsamp	kflux
	aepochSig	upsamp	kepochSig*2
	aepochRms	upsamp	kepochRms*3.5
	aepochZCcps	upsamp	kepochZCcps/500

	fout "../analysis_test.wav", 14, a1, atrig1, \
					astatus,\
					astatusTrig,\
					arms1gated,\
					arms1,\
					acps1,\
					acps1p,\
					aautocorr,\
					acentroid,\
					aspread,\
					askewness,\
					aurtosisM,\
					aflatness,\
					acrest,\
					aflux,\
					aepochSig,\
					aepochRms,\
					aepochZCcps
endif
*/
	endin

