	iwtype 		= 1
	fsin 		pvsanal	a1, gifftsize, gifftsize/4, gifftsize, iwtype
	kEqOn		chnget "selfAutoEqEnable"
	kflag   	pvsftw	fsin,gifnaEq      	        	; export  amps  and freqs to table
    ieqRange    chnget "autoEqRange"
    print ieqRange
    ieqAmount   = .02/ieqRange
				vmult_i  gifnaEqProfile, ieqAmount, giFftTabSize-1

	if kflag == 0 kgoto contin		   		            ; only proc when frame is ready
			    vsubv  gifnaEq, gifnaEqProfile , giFftTabSize-1	; subtract EQ profile 
			    vlimit gifnaEq, 0, 1,  giFftTabSize-1		; limit
			    pvsftr fsin, gifnaEq
contin:
	aout        pvsynth fsin
	a1          delay a1, gifftsize/sr
	a1          = (aout*kEqOn)+ (a1*(1-kEqOn))

