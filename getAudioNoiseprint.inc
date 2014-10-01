; get noise print
	Smess1		= "Measuring spectral profile of background noise, KEEP QUIET!"
			puts, Smess1, 1
	ain		chnget "in1"
	iwtype 		= 1
	fsin 		pvsanal	ain, gifftsize, gifftsize/4, gifftsize, iwtype
	kflag   	pvsftw	fsin,gifnaNoiseIn  	        	

	; copy pvs data from table to array
	kArrA[]  		init    giFftTabSize
	kArrAprev[]  		init    giFftTabSize
	kArrAfilt[]  		init    giFftTabSize

if (kflag > 0) then
	kArrAprev[]		= kArrA
        			copyf2array kArrA, gifnaNoiseIn
	kindx 			= 0
	kfiltered		= 0
  process:
	kaverage		= (kArrA[kindx]+kArrAprev[kindx])*0.5
	kArrAfilt[kindx]	= (kaverage*0.1)+(kArrAfilt[kindx]*0.9)
	kindx 			= kindx + 1
  if kindx < giFftTabSize-2 then
  kgoto process
  endif
				copya2ftab kArrAfilt, gifnaNoise
endif
