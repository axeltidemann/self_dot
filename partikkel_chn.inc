/*
; default
	kamp		= 1		; amp in dB
	kGrainRate	= 10		; number of grains per second
	kGrainDur	= 1		; length of each grain relative to grain rate 
	ksustain_amount	= 0		; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
	ka_d_ratio	= 0.5 		; balance between attack time and decay time, 0.0 = zero attack time and full decay time
	kwavfreq	= 220		; transposition factor (playback speed) of audio inside grains, 
	kgainOct	= 0		; octaviation, amp mod on every 2nd grain

	kdevAmount	= 0		; synchronous/asynchronous balance
	kGrFmFreq	= 10		; FM freq for modulating the grainrate 
	kGrFmIndex	= 0.0		; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
	kdistribution	= 0.0		; grain random distribution in time
	krandommask	= 0		; stochastic masking

	kwavekey1	= 1		; individual source pitch
	kwavekey2	= 1		; individual source pitch
	kwavekey3	= 1		; individual source pitch
	kwavekey4	= 1		; individual source pitch
	kPtchFmFreq	= 440		; FM freq, modulating waveform pitch
	kPtchFmIndex	= 0		; FM index, modulating waveform pitch

	knumpartials	= 4		; number of partials in trainlet
	kchroma		= 2		; chroma, falloff of partial amplitude towards sr/2
	kwavemorf	= 0		; wave mix morfing (range 0 to 3)
*/

; read values
	kamp		chnget "partikkel1_amp"
	kGrainRate	chnget "partikkel1_grainrate"	
	kGrainDur	chnget "partikkel1_graindur"
	ksustain_amount	chnget "partikkel1_sustain"
	ka_d_ratio	chnget "partikkel1_adratio"
	kwavfreq	chnget "partikkel1_wavfreq"
	kgainOct	chnget "partikkel1_octaviation"

	kdevAmount	chnget "partikkel1_async_amount"
	kdistribution	chnget "partikkel1_distribution"
	krandommask	chnget "partikkel1_randomask"
	kGrFmFreq	chnget "partikkel1_grFmFreq"
	kGrFmIndex	chnget "partikkel1_grFmIndex"

	kwavekey1	chnget "partikkel1_wavekey1"
	kwavekey2	chnget "partikkel1_wavekey2"
	kwavekey3	chnget "partikkel1_wavekey3"
	kwavekey4	chnget "partikkel1_wavekey4"
	kPtchFmFreq	chnget "partikkel1_pitchFmFreq"
	kPtchFmIndex	chnget "partikkel1_pitchFmIndex"

	knumpartials	chnget "partikkel1_trainPartials"
	kchroma		chnget "partikkel1_trainChroma"
	kwavemorf	chnget "partikkel1_wavemorf"

; range control
	kamp		limit 	kamp, 0, 1		
	kGrainRate	limit   kGrainRate, 1, 1000	
	kGrainDur	limit	kGrainDur, 0.1, 4
	ksustain_amount	limit	ksustain_amount, 0, 1
	ka_d_ratio	limit	ka_d_ratio, 0, 1
	kwavfreq	limit	kwavfreq, 30, 1500
	kgainOct	limit	kgainOct, 0, 1

	kdevAmount	limit	kdevAmount, 0, 1
	kdistribution	limit	kdistribution, 0, 1
	krandommask	limit	krandommask, 0, 1
	kGrFmFreq	limit	kGrFmFreq, 1, 1000	
	kGrFmIndex	limit	kGrFmIndex, 0, 10

	kwavekey1	limit	kwavekey1, 0.25, 4
	kwavekey2	limit	kwavekey2, 0.25, 4	
	kwavekey3	limit	kwavekey3, 0.25, 4	
	kwavekey4	limit	kwavekey4, 0.25, 4
	kPtchFmFreq	limit	kPtchFmFreq, 1, 1000	
	kPtchFmIndex	limit	kPtchFmIndex, 0, 10	

	knumpartials	limit	knumpartials, 1, 20	
	kchroma		limit	kchroma, 1, 10		
	kwavemorf	limit	kwavemorf, 0, 3
	
; provide critical defaults if channel empty
	;if kamp == 0 then
	;		chnset 1, "partikkel1_amp"		
	;endif
	if kGrainRate == 0 then
			chnset 10, "partikkel1_grainrate"	
	endif
	if kGrainDur == 0 then
			chnset 1, "partikkel1_graindur"
	endif
	if kwavfreq == 0 then
			chnset 1, "partikkel1_wavfreq"
	endif
	if kwavekey1 == 0 then 	
			chnset 1, "partikkel1_wavekey1"
	endif
	if kwavekey2 == 0 then 	
			chnset 1, "partikkel1_wavekey2"
	endif
	if kwavekey3 == 0 then 	
			chnset 1, "partikkel1_wavekey3"
	endif
	if kwavekey4 == 0 then 	
			chnset 1, "partikkel1_wavekey4"
	endif
