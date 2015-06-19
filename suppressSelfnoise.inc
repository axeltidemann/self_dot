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

	ain		chnget "in1"
	ilatency	chnget "audio_io_latency"
	ilatency	limit ilatency-0.01, 1/kr, 1
	aout		chnget "MyOutput"
	aout		delay aout, ilatency
        aoutHP          butterhp aout, 4000
        aout            = (aout*1.2) + (aoutHP*7) ; increase the amount of HF in the feedback signal as it seems they are harder to cancel out
	iwtype 		= 1
	fsin 		pvsanal	ain, gifftsize, gifftsize/4, gifftsize, iwtype
	kreduce		chnget "selfSpectralSuppress"
	fsout 		pvsanal	aout*kreduce, gifftsize, gifftsize/4, gifftsize, iwtype
	;fsouts 		pvsmooth fsout, 0.5, 0.5		
	kflag   	pvsftw	fsin,gifnaIn  	        	; export  amps  and freqs to table
	kflag1   	pvsftw	fsout,gifnaOut	        	; export  amps  and freqs to table
	iNPscale	= ampdbfs(-3)
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
