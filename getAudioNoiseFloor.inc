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
end:
	kdB	= dbfsamp(krmsavg)
	knfloor	= ampdbfs(kdB+8)
	Slevel	sprintfk "Noise floor set to %f dB, over measured %f dB, compare filtered %f", dbfsamp(knfloor), kdB, dbfsamp(krmsF)
		puts Slevel, krmsavg
		chnset krmsavg, "inputNoisefloor"
