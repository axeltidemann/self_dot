
; init chn values for various purposes

        chnset 2000, "td_HiShelfFq"     ; pre emphasis for transient detection of sibliants
        chnset 9, "td_HiShelfGain"     ; in dB
        chnset 200, "	"
        chnset -12, "td_LoShelfGain"
        ;;;NO!;chnset -10, "td_AmpAdjust"      ; approximate amp compensation for EQ
        
        chnset -60, "inputNoisefloor"
        chnset 0, "musicMode"


        ; transient detect parameters
        chnset 1.7, "in_envelopeRelease"    ; envelope follower release
        chnset 5, "transientThresh"         ; transient trig threshold 
        chnset 8, "transientDecThresh"       ; retrig threshold, how much must the level decay from its local max before allowing new transient trig
        chnset 0.7, "doubleLimit"       ; minimum duration between transients, (double trig limit)
        chnset 9, "statusThresh"            ; release thresh for audio status
        chnset 0.8, "statusRel"               ; status release time, hold status=1 for this long after signal has dropped below

        chnset 1.0, "selfSpectralSuppress"        ; amount of FFT analysis of self output subtracted from input, if instr 16 is active

        ; these three are active only for the "secondary synth" voices, and for send between effects (i.e. only the send from delay to reverb)
        chnset -1, "SecondaryDryLevel"          ; in dB
        chnset -14, "SecondaryDelaySendLevel"    ; in dB
        chnset -14, "SecondaryReverbSendLevel"   ; in dB ; affects send from delay to reverb

        chnset 0.8, "SecondaryDelayTime"
        chnset 0.3, "SecondaryDelayFeed"
        chnset 1500, "SecondaryDelayFiltFq"
        chnset 0.91, "SecondaryReverbFeed"
        chnset 6000, "SecondaryReverbFiltFq"

        chnset 450, "SecondaryMasterLoCut"  ;300     ; cut LF on master out for secondary associations synth
        chnset -7, "SecondaryMasterLevel"       ; in dB
        chnset -18, "SecondaryCompThresh" ;-12
        chnset 3, "SecondaryCompRatio"

        chnset -2, "PrimaryMasterLevel"       ; in dB
        chnset -9, "PrimaryCompThresh"
        chnset 3, "PrimaryCompRatio"
        
	    chnset 1, "selfAutoEqEnable"

        chnset -3, "ambientLevel"      ; set level of self's own background sound


	chnset  6000, "Master1HiShelfFq" ; 9000
	chnset  0, "Master1HiShelfGain" ; 1
	chnset  180, "Master1LoShelfFq"
	chnset  -2, "Master1LoShelfGain" 
	chnset  3000, "Master1Band1Fq" ;2000
	chnset  -3, "Master1Band1Gain" ; -2
	chnset  0.6, "Master1Band1Q"
	
	chnset  7000, "Master2HiShelfFq" ;8000
	chnset  -3, "Master2HiShelfGain" ;1
	chnset  250, "Master2LoShelfFq"
	chnset  -9, "Master2LoShelfGain" 
	chnset  5500, "Master2Band1Fq" ; 3500
	chnset  -5, "Master2Band1Gain" ;-3
	chnset  0.6, "Master2Band1Q"
	

	
