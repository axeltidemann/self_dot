; compressor
/*
	kthresh		= p5				; compression threshold
	kratio		= p6				; compression ratio
	kattack		= p7				; attack time 
	krelease	= p8				; release time 
*/
	arms		follow2 a1, kattack/1000, krelease		; envelope follower
	karms		downsamp arms						    ; workaround for dbfsamp (should take a-rate input)
	arms_dB		= dbfsamp(karms)					    ; convert to dB scale
	aovershoot	= arms_dB - kthresh 					; how much over the threshold are we?
    kinvratio   divz 1, kratio, 1
	atarget		= kthresh + (aovershoot*kinvratio)		; target output level (with current input level, threshold and ratio)
	ampMod_dB	=  atarget - arms_dB					; difference from target = adjust amount
	ampMod_dB	limit ampMod_dB, -150, 0				; do not adjust unless negative
	ampMod		= ampdbfs(ampMod_dB)					; convert back to normalized scale
	a1		    = a1*ampMod						        ; apply amplitude modification

