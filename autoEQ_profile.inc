; get Eq profile
	Smess1		= "Measuring audio system EQ profile, KEEP QUIET!"
			    puts, Smess1, 1
	iwtype 		= 1
	fsin 		pvsanal	a1, gifftsize, gifftsize/4, gifftsize, iwtype
	kflag   	pvsftw	fsin,gifnaEq_test  	        	

	; copy pvs data from table to array
	kArrA[]  		init    giFftTabSize
	kArrAprev[]  	init    giFftTabSize
	kArrAfilt[]  	init    giFftTabSize

if (kflag > 0) then
	kArrAprev[]		= kArrA
        			copyf2array kArrA, gifnaEq_test
	kindx 			= 0
	kfiltered		= 0
  process:
	kaverage		= (kArrA[kindx]+kArrAprev[kindx])*0.5
	kArrAfilt[kindx]= (kaverage*0.1)+(kArrAfilt[kindx]*0.9)
	kindx 			= kindx + 1
  if kindx < giFftTabSize-2 then
  kgoto process
  endif
				copya2ftab kArrAfilt, gifnaEqProfile
endif
	kMin_1      minarray kArrAfilt
	kMax_1, kmidx      maxarray kArrAfilt
	printk2 kmidx
	printk2 kMax_1, 10
                chnset kMax_1, "autoEqRange"
    anoise      rnd31 0.8, 1
                outch 1, anoise
