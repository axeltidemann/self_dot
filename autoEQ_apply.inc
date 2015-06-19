;    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
;
;    This file is part of [self.]
;
;    [self.] is free software: you can redistribute it and/or modify
;    it under the terms of the GNU General Public License version 3 
;    as published by the Free Software Foundation.
;
;    [self.] is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with [self.].  If not, see <http://www.gnu.org/licenses/>.


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

