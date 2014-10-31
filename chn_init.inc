
; init chn values for various purposes

	chnset 1000, "td_HiShelfFq"	; pre emphasis for transient detection of sibliants
	chnset 20, "td_HiShelfGain"	; in dB
	chnset 200, "td_LoShelfFq"
	chnset -20, "td_LoShelfGain"
	chnset -15, "td_AmpAdjust"	; approximate amp compensation for EQ
	
	; transient detect parameters
	chnset 2.0, "in_envelopeRelease"    ; envelope follower release
    chnset 4, "transientThresh"         ; transient trig threshold 
	chnset 8, "transientDecThresh"       ; retrig threshold, how much must the level decay from its local max before allowing new transient trig

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
	chnset -2, "SecondaryMasterLevel"       ; in dB
	chnset -18, "SecondaryCompThresh"
	chnset 5, "SecondaryCompRatio"

	chnset -2, "PrimaryMasterLevel"       ; in dB
	chnset -12, "PrimaryCompThresh"
	chnset 4, "PrimaryCompRatio"

