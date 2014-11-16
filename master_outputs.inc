; ******************************
; background sound (instr 90, 91)
; this will be playing as an ambient wash in the room,
; and it will be dampened when self is listening to input (audioStatus).
; As such, it provides feedback to the person speaking to self that it is listening,
; in addition to creating a suitable sonic atmosphere

#include "partikkel_ambient_ftables.inc"	

	instr 92
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
;                        chnmix aoutL, "MasterOut1"
;                        chnmix aoutR, "MasterOut2"
                        outs aoutL, aoutR
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

	a1	        chnget "MasterOut1"
        a1              butterhp a1, 100                ; low cut 

	kHiShelfFq      chnget  "Master1HiShelfFq"; 4000
	kHiShelfGain    chnget  "Master1HiShelfGain"; = 0
	a1	        pareq a1, kHiShelfFq, ampdb(kHiShelfGain), 0.8,  2
	kLoShelfFq      chnget  "Master1LoShelfFq"; = 100
	kLoShelfGain    chnget  "Master1LoShelfGain" ; = 0
	a1	        pareq a1, kLoShelfFq, ampdb(kLoShelfGain), 0.8,  1
	kBand1Fq        chnget  "Master1Band1Fq"; = 1200
	kBand1Gain      chnget  "Master1Band1Gain" ; = -20
	kBand1Q         chnget  "Master1Band1Q"; ; = 0.3
	a1	        pareq a1, kBand1Fq, ampdb(kBand1Gain), kBand1Q,  0

	kthresh         chnget "PrimaryCompThresh"
	kratio          chnget "PrimaryCompRatio"
	kattack		= 15				; attack time 
	krelease	= 20				; release time 
#include "compressor.inc"
	klevel          chnget "PrimaryMasterLevel"
        klevel          = ampdbfs(klevel)
        a1out           = a1*klevel


        a2d             chnget "SecondaryDry"
	a2m	        chnget "MasterOut2"
        a2              = a2d+a2m
        kLoFreq         chnget "SecondaryMasterLoCut"
        a2              butterhp a2, kLoFreq        
        a2              butterhp a2, kLoFreq*0.8        

	kHiShelfFq      chnget  "Master2HiShelfFq"; 4000
	kHiShelfGain    chnget  "Master2HiShelfGain"; = 0
	a2	        pareq a2, kHiShelfFq, ampdb(kHiShelfGain), 0.8,  2
	kLoShelfFq      chnget  "Master2LoShelfFq"; = 100
	kLoShelfGain    chnget  "Master2LoShelfGain" ; = 0
	a2	        pareq a2, kLoShelfFq, ampdb(kLoShelfGain), 0.8,  1
	kBand1Fq        chnget  "Master2Band1Fq"; = 1200
	kBand1Gain      chnget  "Master2Band1Gain" ; = -20
	kBand1Q         chnget  "Master2Band1Q"; ; = 0.3
	a2	        pareq a2, kBand1Fq, ampdb(kBand1Gain), kBand1Q,  0

        kthresh         chnget "SecondaryCompThresh"
        kratio          chnget "SecondaryCompRatio"
        kattack         = 15                                ; attack time 
        krelease        = 20                                ; release time 
        a1              = a2 ; patch into compressor
#include "compressor.inc"
        a2              = a1 ; patch out of compressor
        klevel          chnget "SecondaryMasterLevel"
        klevel          = ampdbfs(klevel)
        a2out           = a2*klevel

        iMasterAdjust   = 2
	                outs a1out*iMasterAdjust, a2out*iMasterAdjust
		        chnset (a1out)*0.5, "MyOutput"

        ; reset channels 
        a0              = 0
                        chnset a0, "SecondaryDry"
                        chnset a0, "SecondaryDelaySend"
                        chnset a0, "SecondaryReverbSend"
                        chnset a0, "MasterOut1"
                        chnset a0, "MasterOut2"
                        chnset a0, "in1"
                        chnset a0, "in2"
                        chnset a0, "in1clean"
                        chnset a0, "in2clean"
	endin
