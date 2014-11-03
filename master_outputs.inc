; ******************************
; background sound (instr 90, 91)
; this will be playing as an ambient wash in the room,
; and it will be dampened when self is listening to input (audioStatus).
; As such, it provides feedback to the person speaking to self that it is listening,
; in addition to creating a suitable sonic atmosphere

    instr 90
; load ftables for instr 91
#include "partikkel_ambient_ftables.inc"	
    endin

	instr 91
; synthesize background sound	
#include "partikkel_ambient.inc"	

	kaLevel     chnget "ambientLevel"
    klevel      = ampdbfs(kaLevel)
    kstatus     chnget "audioStatus"
    kstatus     = 1-kstatus
    alevel      interp kstatus
    afollow     follow2 alevel*a1, 0.1, 3.5
    a1          = a1*klevel*afollow
    a2          = a2*klevel*afollow
                chnmix a1, "ReverbLeft"
                chnmix a2, "ReverbRight"

    endin
    
;******************************************************
        instr 93

        ; reverb for ambience
        aL              chnget "ReverbLeft"
        aR              chnget "ReverbRight"
        a0              = 0
                        chnset a0, "ReverbLeft"
                        chnset a0, "ReverbRight"

        kfb             = 0.87;0.93
        kfco            = 7000
        iverb1Lev       = 0.2
        averb1L,averb1R reverbsc aL, aR, kfb, kfco

        ksize           = 0.8;0.9
        kHF             = 0.2
        iverb2Lev       = 0.6
        averb2L,averb2R freeverb aL, aR, ksize, kHF

        iMaster         = 0.5
        aoutL           = (aL+(averb1L*iverb1Lev)+(averb2L*iverb2Lev))*iMaster
        aoutR           = (aR+(averb1R*iverb1Lev)+(averb2R*iverb2Lev))*iMaster
                        chnmix aoutL, "MasterOut1"
                        chnmix aoutR, "MasterOut2"
 
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
