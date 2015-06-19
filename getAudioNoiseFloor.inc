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


; test signal level
	Smess 	sprintf "Calibrating input signal noise floor, keep relatively quiet for %i seconds", p3
		puts Smess, 1

	a1	chnget "in1"
	krms	rms a1
	krmsF	tonek krms, 1
	kRMS[]	init kr*p3
	kndex	init 0
	kx	init 0
	kRMS[kndex] = krms
	ksum 	init 0
	kmin	init 0
	kmax	init 0
	kflag	init 0
	krmsavg	init 0
if kndex >= (kr*p3)-2 then
	kmin	minarray kRMS
	kmax	maxarray kRMS
	krange	= kmax-kmin
	kflag	= 1
endif
	kndex	= kndex + 1
	kx	= 0
if kflag == 0 goto end
	quartile:
  if kx < lenarray(kRMS) then
 	 k1	= kRMS[kx]
	 k1	limit k1, kmin+(krange*0.25), kmin+(krange*0.75)	; interquartile range
	 kRMS[kx] = k1
	 kx	= kx + 1
	 kgoto quartile
  endif
	ksum	sumarray kRMS
	krmsavg	= ksum / lenarray(kRMS)

	kdB	= dbfsamp(krmsavg)
	knfloor	= kdB+9
	Slevel	sprintfk "Noise floor set to %f dB, over measured %f dB, compare filtered %f", knfloor, kdB, dbfsamp(krmsF)
		puts Slevel, abs(knfloor)
		chnset knfloor, "inputNoisefloor"
end:
