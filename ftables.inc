; pvs ftables
	gifftsize 	= 512
			chnset gifftsize, "fftsize"
	giFftTabSize	= (gifftsize / 2)+1
	gifna     	ftgen   1 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis
	gifnf     	ftgen   2 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis

	gifnaSelf     	ftgen   4 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of my own output
	gifnfSelf     	ftgen   5 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of my own output

	gifnaResyn     	ftgen   7 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs resynthesis
	gifnfResyn     	ftgen   8 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs resynthesis

	gifnaIn     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of input (self output suppression)
	gifnaOut     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of output (self output suppression)

	giNoiseFloor	ftgen 0, 0, 8192, 2, 0					; just init, to be used as noise gate 
	gifnaNoiseIn   	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of bacground noise
	gifnaNoise     	ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis of bacground noise

; classic waveforms
	giSine		ftgen	0, 0, 65536, 10, 1					; sine wave
	giCosine	ftgen	0, 0, 8192, 9, 1, 1, 90					; cosine wave
	giTri		ftgen	0, 0, 8192, 7, 0, 2048, 1, 4096, -1, 2048, 0		; triangle wave 

; grain envelope tables
	giSigmoRise 	ftgen	0, 0, 8193, 19, 0.5, 1, 270, 1				; rising sigmoid
	giSigmoFall 	ftgen	0, 0, 8193, 19, 0.5, 1, 90, 1				; falling sigmoid
	giExpFall	ftgen	0, 0, 8193, 5, 1, 8193, 0.00001				; exponential decay
	giTriangleWin 	ftgen	0, 0, 8193, 7, 0, 4096, 1, 4096, 0			; triangular window 

