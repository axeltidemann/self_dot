;*******************************
; setup of source waveforms
; (needs to be done first, because grain pitch and time pointer depends on source waveform lengths)
;*******************************

; source waveform selection automation
	kwaveform1	= iSound
	kwaveform2	= iSound
	kwaveform3	= iSound
	kwaveform4	= iSound
	
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
	kwavekey1	= 1		; individual source pitch
	kwavekey2	= 1.01		; individual source pitch
	kwavekey3	= 0.99		; individual source pitch
	kwavekey4	= 1.02		; individual source pitch

; set original key dependant on waveform length 
	kwavekey1	= kwavekey1/kfildur1
	kwavekey2	= kwavekey2/kfildur2
	kwavekey3	= kwavekey3/kfildur3
	kwavekey4	= kwavekey4/kfildur4

	kTimeRate	= 0.6				; time pointer rate

	;asamplepos0	phasor kTimeRate / kfildur1	; phasor from 0 to 1, scaled to the length of the first source waveform
        ;idur_once       = 1/ (i(kTimeRate) / i(kfildur1))
        ;print idur_once
        asamplepos0	line 0, p3, 1
	ksamplepos0     downsamp asamplepos0
        ;printk2 ksamplepos0, 20
        asamplepos1     init 0
	kTimeRate2	= 0.2				; segment time pointer rate
        ichangeRate     = 2
        kmetro          metro ichangeRate
        ;asposSH         samphold asamplepos0, kmetro
        kcount          init -1
        kcount          += kmetro
        kcount          = kcount%4
        printk2 kcount
        kcpsamp1        init 0
        kcpsamp2        init 0
        kcpsamp3        init 0
        kcpsamp4        init 0
        if kmetro > 0 then
          if kcount == 0 then
            kcpsamp1    = ichangeRate*0.25
            reinit spos1
          elseif kcount == 1 then
            kcpsamp2    = ichangeRate*0.25
            reinit spos2
          elseif kcount == 2 then
            kcpsamp3    = ichangeRate*0.25
            reinit spos3
          elseif kcount == 3 then
            kcpsamp4    = ichangeRate*0.25
            reinit spos4
          endif
        endif
/*
        if kcount == 0 then
        kgoto spos1
        elseif kcount == 1 then
        kgoto spos2
        elseif kcount == 2 then
        kgoto spos3
        elseif kcount == 3 then
        kgoto spos4
        endif
*/
spos1:
        ks1amp          oscil1, 0, 1, 1/i(kcpsamp1), giHanning
        aspos1          phasor kTimeRate2 / kfildur1
        ipos1           = i(ksamplepos0)
        asamplepos1     = ipos1 + aspos1
        ;agate           table  asamplepos1, giSampleposGate
        ;kagate          downsamp agate
        ;kspos1 downsamp aspos1
        ;printk2 kspos1, 20
        ;ks1amp          = ks1amp*kagate
        rireturn
        ;kgoto donespos
spos2:
        ks2amp          oscil1, 0, 1, 1/i(kcpsamp2), giHanning
        aspos2          phasor kTimeRate2 / kfildur1
        ipos2           = i(ksamplepos0)
        asamplepos2     = ipos2 + aspos2
        rireturn
        ;kgoto donespos
spos3:
        ks3amp          oscil1, 0, 1, 1/i(kcpsamp3), giHanning
        aspos3          phasor kTimeRate2 / kfildur1
        ipos3           = i(ksamplepos0)
        asamplepos3     = ipos3 + aspos3
        rireturn
        ;kgoto donespos
spos4:
        ks4amp          oscil1, 0, 1, 1/i(kcpsamp4), giHanning
        aspos4          phasor kTimeRate2 / kfildur1
        ipos4           = i(ksamplepos0)
        asamplepos4     = ipos4 + aspos4
        rireturn
        ;kgoto donespos
donespos:

; samplepos rDev
        ksampRdevAmt    = 0.001/kfildur1
        ksampRdev1      randh ksampRdevAmt, 30
        ksampRdev2      randh ksampRdevAmt, 30
        ksampRdev3      randh ksampRdevAmt, 30
        ksampRdev4      randh ksampRdevAmt, 30
        asamplepos1     = asamplepos1+ksampRdev1
        asamplepos2     = asamplepos2+ksampRdev2
        asamplepos3     = asamplepos3+ksampRdev3
        asamplepos4     = asamplepos4+ksampRdev4

	; wave mix masking. 
	; Set gain per source waveform per grain, 
	; in groups of 5 amp values, reflecting source1, source2, source3, source4, and the 5th slot is for trainlet amplitude.
	iwaveamptab	ftgentmp	0, 0, 32, -2, 0, 0,   0,0,0,0,0
        ks1amp          init 0
        ks2amp          init 0
        ks3amp          init 0
        ks4amp          init 0
        
                        tablew ks1amp, 2, iwaveamptab
                        tablew ks2amp, 3, iwaveamptab
                        tablew ks3amp, 4, iwaveamptab
                        tablew ks4amp, 5, iwaveamptab


;*******************************
; other granular synthesis parameters
;*******************************

; amplitude
	kamp		= ampdbfs(-6)				; output amplitude

; grain rate
	kGrainRate	= 120					; number of grains per second

; sync 
	kdevAmount	= 0.3					; synchronous/asynchronous balance
	async		probabilityClock kGrainRate		; asynchronous clock
	async		= async*kdevAmount			; Balance the weight of the sync pulse
	agrainrate	= kGrainRate*(1-kdevAmount)		; ... with the rate of the internal clock

; grain rate FM
	kGrFmFreq	= 10					; FM freq for modulating the grainrate 
	kGrFmIndex	= 0.0					; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
	iGrFmWave	= giSine				; FM waveform, for modulating the grainrate 
	aGrFmSig	oscil kGrFmIndex, kGrFmFreq, iGrFmWave	; audio signal for frequency modulation of grain rate
	agrainrate	= kGrainRate + (aGrFmSig*kGrainRate)	; add the modulator signal to the grain rate signal

; distribution 
	kdistribution	= 0.3						; grain random distribution in time
	idisttab	ftgentmp	0, 0, 16, 16, 1, 16, -10, 0	; probability distribution for random grain masking

; grain shape
        kGrainDur	= 4					; length of each grain relative to grain rate 
	kduration	= (kGrainDur*1000)/kGrainRate		; grain dur in milliseconds, relative to grain rate

	ienv_attack	= giSigmoRise 				; grain attack shape (from table)
	ienv_decay	= giSigmoFall 				; grain decay shape (from table)
	ksustain_amount	= 0					; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
	ka_d_ratio	= 0.5 					; balance between attack time and decay time, 0.0 = zero attack time and full decay time
	kenv2amt	= 0					; amount of secondary enveloping per grain (e.g. for fof synthesis)
	ienv2tab	= giExpFall 				; secondary grain shape (from table), enveloping the whole grain if used

; grain pitch (transpose, or "playback speed")
	kwavfreq	= 1					; transposition factor (playback speed) of audio inside grains, 

; pitch sweep
	ksweepshape		= 0.75
	iwavfreqstarttab 	ftgentmp	0, 0, 16, -2, 0, 0,   1		; start freq scalers, per grain
	iwavfreqendtab		ftgentmp	0, 0, 16, -2, 0, 0,   1		; end freq scalers, per grain

; FM of grain pitch (playback speed)
	kPtchFmFreq	= 440							; FM freq, modulating waveform pitch
	kPtchFmIndex	= 0							; FM index, modulating waveform pitch
	iPtchFmWave	= giSine						; FM waveform, modulating waveform pitch
	ifmamptab	ftgentmp	0, 0, 16, -2, 0, 0,   1			; FM index scalers, per grain
	ifmenv		= giTriangleWin 					; FM index envelope, over each grain (from table)
	kPtchFmIndex	= kPtchFmIndex + (kPtchFmIndex*kPtchFmFreq*0.00001) 	; FM index scaling formula
	awavfm		oscil	kPtchFmIndex, kPtchFmFreq, iPtchFmWave		; Modulator signal for frequency modulation inside grain

; trainlet parameters
	icosine		= giCosine				; needs to be a cosine wave to create trainlets
	kTrainCps	= kGrainRate/kGrainDur			; create a single cycle of a trainlet inside each grain
	knumpartials	= 4					; number of partials in trainlet
        kchroma		= 2					; chroma, falloff of partial amplitude towards sr/2

; masking
	; gain masking table, amplitude for individual grains
	igainmasks	ftgentmp	0, 0, 16, -2, 0, 1,   1, 1

	kgainOct	= 0					; octaviation, amp mod on every 2nd grain
			tablew	1-sqrt(kgainOct), 3, igainmasks	; write modified gain mask, every 2nd grain will get a modified amplitude 


	; channel masking table, output routing for individual grains (zero based, a value of 0.0 routes to output 1)
	ichannelmasks	ftgentmp	0, 0, 16, -2,  0, 3,  0.5, 0.5, 0.5, 0.5
	
	; random masking (muting) of individual grains
	krandommask	= 0					; stochastic masking, masking off for the first 7 seconds, then increase stochastic masking

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
        aout            = a1+a2
        aout            butterhp aout, 300
        aout            butterhp aout, 400


