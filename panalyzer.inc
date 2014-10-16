
	aLeft	chnget "in1clean" ; use the non-ducked input signal
	aRight	chnget "in2clean"

; preprocess
/*
    aLeft           butterhp aLeft, 400
    aRight          butterhp aRight, 400
    aLeft           butterhp aLeft, 300
    aRight          butterhp aRight, 300
*/
; amplitude comparision
	krmsLg		    rms aLeft
	krmsRg		    rms aRight
	armsL           follow2 aLeft, 0.01, 0.5
	armR            follow2 aRight, 0.01, 0.5
	krmsL           downsamp armsL
	krmsR           downsamp armR
	ktempdiv	    divz (krmsL - krmsR), (krmsL+krmsR), 1
	krmsPos		    = ((ktempdiv)+1)*0.5
	krmsPosLarge	= ((ktempdiv)*3)

; gate /samphold
	kgate		    = -22
	kgateL		    = (dbfsamp(krmsLg) > kgate ? 1 : 0)
	kgateR		    = (dbfsamp(krmsRg) > kgate ? 1 : 0)
	
; ***************
;analyze transients
	kAttack		= 0.001				        ; envelope follower attack
	kRelease	= 0.3 				        ; envelope follower release
	aFollow1	follow2	(aLeft+aRight)*0.5, kAttack, kRelease	; envelope follower
	kFollow1	downsamp aFollow1	
	kFollowdb1	= dbfsamp(kFollow1)		    ; convert to dB
	iresponse	= 10 			            ; response time in milliseconds
	ktthresh	= 9			                ; transient trig threshold 
	klowThresh	chnget "inputNoisefloor"	; lower threshold for transient detection
	idoubleLimit	= 0.02			        ; minimum duration between events, (double trig limit)
	kdecThresh	= 6			                ; retrig threshold, how much must the level decay from its local max before allowing new transient trig
	ktransient, kdiff	TransientDetect kFollowdb1, iresponse, ktthresh, klowThresh, kdecThresh, idoubleLimit
	
noDoubleTrig:
    idoublePeriod   = 1
    kdoubleMute     linseg 0, idoublePeriod, 0, 0, 1, 1, 1
    rireturn
    ktransient2     = ktransient*kdoubleMute
    kcenter         = 0.5
    if ktransient2 > 0 then
                    chnset krmsPos, "panalyzer_pan"
                    reinit noDoubleTrig
    else
                    chnset kcenter, "panalyzer_pan"
    endif
    ;ktest   chnget "panalyzer_pan"
    ;printk2 ktest, 30


