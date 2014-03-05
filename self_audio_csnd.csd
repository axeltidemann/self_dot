<CsoundSynthesizer>
<CsOptions>
-odac -iadc -d
</CsOptions>

<CsInstruments>

giSine		ftgen	0, 0, 65536, 10, 1

;**************
	instr 1  
	iamp	= ampdbfs(-5)
	icps	= 220
	koffset	chnget "freq_offset"
  	kcps	= icps + koffset
		chnset kcps, "cps"
	a1 	oscili iamp, kcps, giSine	; sine test tone
  	a2 	oscili iamp, kcps*2, giSine	; sine test tone 2
	;a1,a2	inch 1,2			; for audio in to replace the sine test tones
	f1     	pvsanal a1, 1024, 256, 1024, 1 	; analysis
	f2     	pvsanal a2, 1024, 256, 1024, 1 	; 
         	pvsout  f1, 0 			; write signal to pvs out bus channel 0
         	pvsout  f2, 1			; write signal to pvs out bus channel 0

  		out a1,a2
	endin

</CsInstruments>

<CsScore>
; run for N sec
i1 0 10

</CsScore>

</CsoundSynthesizer>
