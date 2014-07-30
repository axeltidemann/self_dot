
; init chn values for various purposes

	chnset -5, "SecondaryDryLevel"          ; in dB
	chnset -18, "SecondaryDelaySendLevel"    ; in dB
	chnset -24, "SecondaryReverbSendLevel"   ; in dB

        chnset 2.0, "SecondaryDelayTime"
        chnset 0.3, "SecondaryDelayFeed"
        chnset 3000, "SecondaryDelayFiltFq"

        chnset 0.94, "SecondaryReverbFeed"
	chnset 7000, "SecondaryReverbFiltFq"

	chnset 400, "SecondaryMasterLoCut"       ; cut LF on master out for secondary associations synth
	chnset -6, "SecondaryMasterLevel"       ; in dB
