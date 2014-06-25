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
	kamp		chnget "partikkel2_amp"
	kGrainRate	chnget "partikkel2_grainrate"	
	kGrainDur	chnget "partikkel2_graindur"
	ksustain_amount	chnget "partikkel2_sustain"
	ka_d_ratio	chnget "partikkel2_adratio"
	kwavfreq	chnget "partikkel2_wavfreq"
	kgainOct	chnget "partikkel2_octaviation"

	kdevAmount	chnget "partikkel2_async_amount"
	kdistribution	chnget "partikkel2_distribution"
	krandommask	chnget "partikkel2_randomask"
	kGrFmFreq	chnget "partikkel2_grFmFreq"
	kGrFmIndex	chnget "partikkel2_grFmIndex"

	kwavekey1	chnget "partikkel2_wavekey1"
	kwavekey2	chnget "partikkel2_wavekey2"
	kwavekey3	chnget "partikkel2_wavekey3"
	kwavekey4	chnget "partikkel2_wavekey4"
	kPtchFmFreq	chnget "partikkel2_pitchFmFreq"
	kPtchFmIndex	chnget "partikkel2_pitchFmIndex"

	knumpartials	chnget "partikkel2_trainPartials"
	kchroma		chnget "partikkel2_trainChroma"
	kwavemorf	chnget "partikkel2_wavemorf"

	kDryLevel	chnget "partikkel2_dryLevel"
	kDelaySend	chnget "partikkel2_delaySend"
	kReverbSend	chnget "partikkel2_reverbSend"

/*
; range control DISABLED for secondary assoc
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
*/
