; self analysis of own output 
	instr 98
	a1	chnget "MasterOut1"
	a2	= 0
/*
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
*/
	endin

; ******************************
; master output 
	instr 99

	a1	chnget "MasterOut1"
	a2	chnget "MasterOut2"
		outs a1, a2
		chnset (a1+a2)*0.5, "MyOutput"

	a0	= 0
		chnset a0, "MasterOut1"
		chnset a0, "MasterOut2"
		chnset a0, "in1"
		chnset a0, "in2"
	endin
