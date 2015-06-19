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
