	ain		chnget "in1"
	ilatency	chnget "audio_io_latency"
	ilatency	limit ilatency-0.01, 1/kr, 1
	aout		chnget "MyOutput"
	aout		delay aout, ilatency
        aoutHP          butterhp aout, 4000
        aout            = (aout*1.2) + (aoutHP*7) ; increase the amount of HF in the feedback signal asa it seems they are harder to cancel out
	iwtype 		= 1
	fsin 		pvsanal	ain, gifftsize, gifftsize/4, gifftsize, iwtype
	ireduce		= p4
	fsout 		pvsanal	aout*ireduce, gifftsize, gifftsize/4, gifftsize, iwtype
	;fsouts 		pvsmooth fsout, 0.5, 0.5		
	kflag   	pvsftw	fsin,gifnaIn  	        	; export  amps  and freqs to table
	kflag1   	pvsftw	fsout,gifnaOut	        	; export  amps  and freqs to table
	iNPscale	= p5
			vmult_i  gifnaNoise, iNPscale, giFftTabSize-1
	if kflag == 0 kgoto contin		   		; only proc when frame is ready
			vsubv  gifnaIn, gifnaOut, giFftTabSize-1	; subtract my own output 
			vsubv  gifnaIn, gifnaNoise, giFftTabSize-1	; subtract general noise profile 
			vlimit gifnaIn, 0, 1,  giFftTabSize-1		; limit

			kindex = 0

	spectralAverage:
			k1	table kindex-2, gifnaIn
			k2	table kindex-1, gifnaIn
			k3	table kindex, gifnaIn
			k4	table kindex+1, gifnaIn
			k5	table kindex+2, gifnaIn
			ksum	= (k1*0.1)+(k2*0.2)+(k3*0.4)+(k4*0.2)+(k5*0.1)
				tablew ksum, kindex, gifnaIn
			kindex = kindex +1
			if kindex < gifnaIn-2 kgoto spectralAverage

			pvsftr fsin, gifnaIn
contin:

	aclean		pvsynth fsin
			chnset aclean, "in1"
			;fout "suppress.wav", 14, ain, aclean
