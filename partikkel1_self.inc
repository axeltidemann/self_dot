;*******************************
; setup of source waveforms
; (needs to be done first, because grain pitch and time pointer depends on source waveform lengths)
;*******************************

; source waveform selection automation
	kwaveform1	= iSound
	kwave1Single	= 0
	kwaveform2	= iSound
	kwave2Single	= 0
	kwaveform3	= iSound
	kwave3Single	= 0
	kwaveform4	= iSound
	kwave4Single	= 0


; get source waveform length (used when calculating transposition and time pointer)
	kfilen1		tableng	 kwaveform1		; get length of the first source waveform
	kfilen2		tableng	 kwaveform2		; same as above, for source waveform 2
	kfilen3		tableng	 kwaveform3		; same as above, for source waveform 3
	kfilen4		tableng	 kwaveform4		; same as above, for source waveform 4
	kfildur1	= kfilen1 / sr			; length in seconds, for the first source waveform
	kfildur2	= kfilen2 / sr			; same as above, for source waveform 2
	kfildur3	= kfilen3 / sr			; same as above, for source waveform 3
	kfildur4	= kfilen4 / sr			; same as above, for source waveform 4

; original pitch for each waveform, use if they should be transposed individually
; can also be used as a "cycles per second" parameter for single cycle waveforms (assuming that the kwavfreq parameter has a value of 1.0)
	;kwavekey1	= 1
	;kwavekey2	= 1
	;kwavekey3	= 1
	;kwavekey4	= 1

; set original key dependant on waveform length (only for sampled waveforms, not for single cycle waves)
	kwavekey1	= (kwave1Single > 0 ? kwavekey1 : divz(kwavekey1,kfildur1,1))
	kwavekey2	= (kwave2Single > 0 ? kwavekey2 : divz(kwavekey2,kfildur2,1))
	kwavekey3	= (kwave3Single > 0 ? kwavekey3 : divz(kwavekey3,kfildur3,1))
	kwavekey4	= (kwave4Single > 0 ? kwavekey4 : divz(kwavekey4,kfildur4,1))

; time pointer (phase). This can be independent for each source waveform.
	isamplepos1	= 0				; initial phase for wave source 1
	isamplepos2	= 0				; initial phase for wave source 2
	isamplepos3	= 0				; initial phase for wave source 3
	isamplepos4	= 0				; initial phase for wave source 4

	kTimeRate	    = 1				; time pointer rate
    asamplepos1     phasor divz(kTimeRate, kfildur1, 1)     ; phasor from 0 to 1, scaled to the length of the first source waveform
    asamplepos2     phasor divz(kTimeRate, kfildur2, 1)     ; same as above, scaled for source wave 2
    asamplepos3     phasor divz(kTimeRate, kfildur3, 1)     ; same as above, scaled for source wave 3
    asamplepos4     phasor divz(kTimeRate, kfildur4, 1)     ; same as above, scaled for source wave 4

	; mix initial phase and moving phase value (moving phase only for sampled waveforms, single cycle waveforms use static samplepos)
	asamplepos1	= asamplepos1*(1-kwave1Single) + isamplepos1
	asamplepos2	= asamplepos2*(1-kwave2Single) + isamplepos2
	asamplepos3	= asamplepos3*(1-kwave3Single) + isamplepos3
	asamplepos4	= asamplepos4*(1-kwave4Single) + isamplepos4

;*******************************
; other granular synthesis parameters
;*******************************

; amplitude
	;kamp		= ampdbfs(-3)				; output amplitude

; grain rate
	;kGrainRate	= 10					; number of grains per second

; sync 
	;kdevAmount	= 0					; synchronous/asynchronous balance
	async		probabilityClock kGrainRate		; asynchronous clock
	async		= async*kdevAmount			; Balance the weight of the sync pulse
	agrainrate	= kGrainRate*(1-kdevAmount)		; ... with the rate of the internal clock

; grain rate FM
	;kGrFmFreq	= 10					; FM freq for modulating the grainrate 
	;kGrFmIndex	= 0.0					; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
	iGrFmWave	= giSine				; FM waveform, for modulating the grainrate 
	aGrFmSig	oscil kGrFmIndex, kGrFmFreq, iGrFmWave	; audio signal for frequency modulation of grain rate
	agrainrate	= kGrainRate + (aGrFmSig*kGrainRate)	; add the modulator signal to the grain rate signal

; distribution 
	;kdistribution	= 0.0						; grain random distribution in time
	idisttab	ftgentmp	0, 0, 16, 16, 1, 16, -10, 0	; probability distribution for random grain masking

; grain shape
	;kGrainDur	= 1					; length of each grain relative to grain rate 
	kduration	= (kGrainDur*1000)/kGrainRate		; grain dur in milliseconds, relative to grain rate

	ienv_attack	= giSigmoRise 				; grain attack shape (from table)
	ienv_decay	= giSigmoFall 				; grain decay shape (from table)
	;ksustain_amount	= 0					; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
	;ka_d_ratio	= 0.5 					; balance between attack time and decay time, 0.0 = zero attack time and full decay time
	kenv2amt	= 0					; amount of secondary enveloping per grain (e.g. for fof synthesis)
	ienv2tab	= giExpFall 				; secondary grain shape (from table), enveloping the whole grain if used

; grain pitch (transpose, or "playback speed")
	;kwavfreq	= 220					; transposition factor (playback speed) of audio inside grains, 

; pitch sweep
	ksweepshape		= 0.75
	iwavfreqstarttab 	ftgentmp	0, 0, 16, -2, 0, 0,   1		; start freq scalers, per grain
	iwavfreqendtab		ftgentmp	0, 0, 16, -2, 0, 0,   1		; end freq scalers, per grain

; FM of grain pitch (playback speed)
	;kPtchFmFreq	= 440							; FM freq, modulating waveform pitch
	;kPtchFmIndex	= 0							; FM index, modulating waveform pitch
	iPtchFmWave	= giSine						; FM waveform, modulating waveform pitch
	ifmamptab	ftgentmp	0, 0, 16, -2, 0, 0,   1			; FM index scalers, per grain
	ifmenv		= giTriangleWin 					; FM index envelope, over each grain (from table)
	kPtchFmIndex	= kPtchFmIndex + (kPtchFmIndex*kPtchFmFreq*0.00001) 	; FM index scaling formula
	awavfm		oscil	kPtchFmIndex, kPtchFmFreq, iPtchFmWave		; Modulator signal for frequency modulation inside grain

; trainlet parameters
	icosine		= giCosine				; needs to be a cosine wave to create trainlets
	kTrainCps	= kGrainRate/kGrainDur			; create a single cycle of a trainlet inside each grain
	;knumpartials	= 16					; number of partials in trainlet
	;kchroma		= 2					; chroma, falloff of partial amplitude towards sr/2

; masking
	; gain masking table, amplitude for individual grains
	igainmasks	ftgentmp	0, 0, 16, -2, 0, 1,   1, 1

	;kgainOct	= 0					; octaviation, amp mod on every 2nd grain
			tablew	1-sqrt(kgainOct), 3, igainmasks	; write modified gain mask, every 2nd grain will get a modified amplitude 


	; channel masking table, output routing for individual grains (zero based, a value of 0.0 routes to output 1)
	ichannelmasks	ftgentmp	0, 0, 16, -2,  0, 3,  0.5, 0.5, 0.5, 0.5
	
	; random masking (muting) of individual grains
	;krandommask	= 0					; stochastic masking, masking off for the first 7 seconds, then increase stochastic masking

	; wave mix masking. 
	; Set gain per source waveform per grain, 
	; in groups of 5 amp values, reflecting source1, source2, source3, source4, and the 5th slot is for trainlet amplitude.
	iwaveamptab	ftgentmp	0, 0, 32, -2, 0, 0,   1,.3,.3,.1,0


; system parameter
	imax_grains	= 100				; max number of grains per k-period

;***********


a1,a2,a3,a4,a5,a6,a7,a8	partikkel agrainrate, kdistribution, idisttab, async, kenv2amt, ienv2tab, \
               	  ienv_attack, ienv_decay, ksustain_amount, ka_d_ratio, kduration, kamp, igainmasks, \
               	  kwavfreq, ksweepshape, iwavfreqstarttab, iwavfreqendtab, awavfm, \
               	  ifmamptab, ifmenv, icosine, kTrainCps, knumpartials, \
               	  kchroma, ichannelmasks, krandommask, kwaveform1, kwaveform2, kwaveform3, kwaveform4, \
               	  iwaveamptab, asamplepos1, asamplepos2, asamplepos3, asamplepos4, \
               	  kwavekey1, kwavekey2, kwavekey3, kwavekey4, imax_grains

; boost some, to balance with other audio generators
a1	= a1*2
a2	= a2*2

