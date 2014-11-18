
; init chn values for various purposes

        chnset 2000, "td_HiShelfFq"     ; pre emphasis for transient detection of sibliants
        chnset 9, "td_HiShelfGain"     ; in dB
        chnset 200, "	"
        chnset -12, "td_LoShelfGain"
        ;;;NO!;chnset -10, "td_AmpAdjust"      ; approximate amp compensation for EQ
        
        chnset -60, "inputNoisefloor"

        ; transient detect parameters
        chnset 1.5, "in_envelopeRelease"    ; envelope follower release
        chnset 5, "transientThresh"         ; transient trig threshold 
        chnset 8, "transientDecThresh"       ; retrig threshold, how much must the level decay from its local max before allowing new transient trig
        chnset 0.7, "doubleLimit"       ; minimum duration between transients, (double trig limit)
        chnset 9, "statusThresh"            ; release thresh for audio status
        chnset 0.8, "statusRel"               ; status release time, hold status=1 for this long after signal has dropped below

        chnset 2.0, "selfSpectralSuppress"        ; amount of FFT analysis of self output subtracted from input, if instr 16 is active

        ; these three are active only for the "secondary synth" voices, and for send between effects (i.e. only the send from delay to reverb)
        chnset -1, "SecondaryDryLevel"          ; in dB
        chnset -14, "SecondaryDelaySendLevel"    ; in dB
        chnset -14, "SecondaryReverbSendLevel"   ; in dB ; affects send from delay to reverb

        chnset 0.8, "SecondaryDelayTime"
        chnset 0.3, "SecondaryDelayFeed"
        chnset 1500, "SecondaryDelayFiltFq"
        chnset 0.91, "SecondaryReverbFeed"
        chnset 6000, "SecondaryReverbFiltFq"

        chnset 300, "SecondaryMasterLoCut"       ; cut LF on master out for secondary associations synth
        chnset -7, "SecondaryMasterLevel"       ; in dB
        chnset -12, "SecondaryCompThresh"
        chnset 3, "SecondaryCompRatio"

        chnset -2, "PrimaryMasterLevel"       ; in dB
        chnset -9, "PrimaryCompThresh"
        chnset 3, "PrimaryCompRatio"
        
	    chnset 1, "selfAutoEqEnable"

        chnset -3, "ambientLevel"      ; set level of self's own background sound


	chnset  9000, "Master1HiShelfFq"
	chnset  3, "Master1HiShelfGain"
	chnset  180, "Master1LoShelfFq"
	chnset  -2, "Master1LoShelfGain" 
	chnset  2000, "Master1Band1Fq"
	chnset  -2, "Master1Band1Gain"
	chnset  0.6, "Master1Band1Q"
	
	chnset  8000, "Master2HiShelfFq"
	chnset  1, "Master2HiShelfGain"
	chnset  250, "Master2LoShelfFq"
	chnset  -9, "Master2LoShelfGain" 
	chnset  3500, "Master2Band1Fq"
	chnset  -3, "Master2Band1Gain"
	chnset  0.6, "Master2Band1Q"
	

	
