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
