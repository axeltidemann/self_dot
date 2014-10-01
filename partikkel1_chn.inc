; partikkel sample playback parameters

	kamp		= 0.1		; amp in dB
	kGrainRate	= 30		; number of grains per second
	kGrainDur	= 2		; length of each grain relative to grain rate 
	ksustain_amount	= 0		; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
	ka_d_ratio	= 0.5 		; balance between attack time and decay time, 0.0 = zero attack time and full decay time
	kwavfreq	= 1		; transposition factor (playback speed) of audio inside grains, 
	kgainOct	= 0		; octaviation, amp mod on every 2nd grain

	kdevAmount	= 0		; synchronous/asynchronous balance
	kGrFmFreq	= 10		; FM freq for modulating the grainrate 
	kGrFmIndex	= 0.0		; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
	kdistribution	= 0.0		; grain random distribution in time
	krandommask	= 0		; stochastic masking

	kwavekey1	= 1		; individual source pitch
	kwavekey2	= 1.5		; individual source pitch
	kwavekey3	= 0.5		; individual source pitch
	kwavekey4	= 1.3		; individual source pitch
	kPtchFmFreq	= 440		; FM freq, modulating waveform pitch
	kPtchFmIndex	= 0.0		; FM index, modulating waveform pitch

	knumpartials	= 4		; number of partials in trainlet
	kchroma		= 2		; chroma, falloff of partial amplitude towards sr/2
	kwavemorf	= 0		; wave mix morfing (range 0 to 3)
