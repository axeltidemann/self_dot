
/*frequency in which the file is read*/
ifildur			= ilen
iwavfreq		= 1 / ifildur
kwavfreq		= iwavfreq * iformant
afilposphas		phasor		ispeed / ifildur

/*read pitch analysis data from ftable*/
;kindx			downsamp afilposphas
;ilast			table	0, iPitchFn
;kindx			= (kindx*ilast)+1
kindx           init 0
kindx           = kindx+ispeed
kcps			table	kindx, iPitchFn
kcps			limit	kcps, 50, 1500

/*other parameters*/
kgrainfreq		= kcps*itransp		; grains per second
kduration		= (1000/kgrainfreq)*1.5	; grain duration ; smoother(?) if *3, but octave down needs duration to be max *2
kdistribution		= 0.1			; periodic grain clock
idisttab	        ftgentmp 0, 0, 16, 16, 1, 16, -10, 0	; probability distribution for random grain masking
ienv2tab		ftgentmp 0, 0, 4096, 20, 9, 1	; sync window for grains 
kenv2amt		= 1			; amount of secondary grain envelope
kamp			= 1			; grain amplitude
async			= 0			; no sync input
ienv_attack		= giSigmoRise		; grain attack envelope
ienv_decay		= giSigmoFall		; grain decay envelope 
ksustain_amount		= 0.5			; grain sustain time (for primary envelope)
ka_d_ratio		= 0.2 			; ratio of attack time to decay time (for primary envelope)

ichannelmasks		= -1                    ; channel masking ftable
igainmasks		= -1			; default (no gain masking)
ksweepshape		= 0			; no frequency sweep
iwavfreqstarttab 	= -1			; default (no frequency sweep start)
iwavfreqendtab		= -1			; default (no frequency sweep end)
awavfm			= 0			; no FM input
ifmamptab		= -1			; default FM scaling (=1)
kfmenv			= -1			; default FM envelope (flat)
krandommask		= 0.0			; no random gain masking
kwaveform1		= iSound		; input sample
kwaveform2		= iSound
kwaveform3		= iSound
kwaveform4		= iSound
iwaveamptab		ftgentmp 0, 0, 32, -2,   0, 0, 		\		; wave mix mask
							1,0,0,0,0,	\
							0,1,0,0,0,	\
							0,0,1,0,0,	\
							0,0,0,1,0	\

; trainlets not used, but parameter values needs to be specified
icosine			= giCosine		; cosine ftable
kTrainCps		= kgrainfreq		; set trainlet cps equal to grain rate for single-cycle trainlet in each grain
knumpartials		= 1			; number of partials in trainlet
kchroma			= 1			; balance of partials in trainlet

/*time pointer*/
iopcode_id		= 1
gatrig			init 0
asamplepos1		= afilposphas
asamplepos2		= asamplepos1+((0.25/kcps) / ifildur)
asamplepos3		= asamplepos1+((0.50/kcps) / ifildur)
asamplepos4		= asamplepos1+((0.75/kcps) / ifildur)

/*original key of sample (default=1)*/
kwavekey1		= 1			; original key
kwavekey2		= kwavekey1*1.3	
kwavekey3		= kwavekey1*0.7
kwavekey4		= kwavekey1
imax_grains		= 100			; maximum number of grains per k-period


a1,a2,a3,a4	partikkel kgrainfreq, kdistribution, idisttab, async, kenv2amt, ienv2tab, \
               	ienv_attack, ienv_decay, ksustain_amount, ka_d_ratio, kduration, kamp, igainmasks, \
               	kwavfreq, ksweepshape, iwavfreqstarttab, iwavfreqendtab, awavfm, \
               	ifmamptab, kfmenv, icosine, kTrainCps, knumpartials, \
               	kchroma, ichannelmasks, krandommask, kwaveform1, kwaveform2, kwaveform3, kwaveform4, \
               	iwaveamptab, asamplepos1, asamplepos2, asamplepos3, asamplepos4, \
               	kwavekey1, kwavekey2, kwavekey3, kwavekey4, imax_grains, iopcode_id

aout		= a1*0.3
;atest           oscili 1, kcps*itransp, giSine
;aout        = aout + atest

