; ******************************
; background sound
; this will be playing as an ambient wash in the room,
; and it will be dampened when self is listening to input (audioStatus).
; As such, it provides feedback to the person speaking to self that it is listening,
; in addition to creating a suitable sonic atmosphere
	instr 91
	
	; white noise to test the idea of sound as monitoring
	a1          rnd31 1, 1
    klevel      = ampdbfs(-5)
    kstatus     chnget "audioStatus"
    ;printk2 kstatus
    alevel      interp kstatus
    afollow     follow2 alevel*a1, 0.1, 1.5
    a1          = 0;kstatus;klevel;*
                chnmix a1, "MasterOut1"
                chnmix a1, "MasterOut2"
                ;outs a1, a1
    endin
    
    
    
; ******************************
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

	kthresh         chnget "PrimaryCompThresh"
	kratio          chnget "PrimaryCompRatio"
	kattack		    = 15				; attack time 
	krelease	    = 20				; release time 
#include "compressor.inc"
	klevel          chnget "PrimaryMasterLevel"
    klevel          = ampdbfs(klevel)
    a1              = a1*klevel

	a2	chnget "MasterOut2"
    iMasterAdjust   = 2
		            outs a1*iMasterAdjust, a2*iMasterAdjust
		            chnset (a1+a2)*0.5, "MyOutput"

	a0	= 0
                    chnset a0, "MasterOut1"
                    chnset a0, "MasterOut2"
                    chnset a0, "in1"
                    chnset a0, "in2"
	endin
