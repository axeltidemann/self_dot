; pvs ftables
	gifftsize 	= 1024
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
	
	gifnaEq_test    ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for audio system Eq profiling
	gifnaEqProfile  ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; init audio system Eq profile
	gifnaEq         ftgen   0 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for audio system Eq application

; classic waveforms
	giSine		ftgen	0, 0, 65536, 10, 1					; sine wave
	giCosine	ftgen	0, 0, 8192, 9, 1, 1, 90					; cosine wave
	giTri		ftgen	0, 0, 8192, 7, 0, 2048, 1, 4096, -1, 2048, 0		; triangle wave 

; grain envelope tables
        giSampleposGate ftgen   0, 0, 8193, 7, 1, 8190, 2, 0                            ; gate at end
	giHanning 	ftgen	0, 0, 8193, 20, 2, 1				        ; Hanning
	giSigmoRise 	ftgen	0, 0, 8193, 19, 0.5, 1, 270, 1				; rising sigmoid
	giSigmoFall 	ftgen	0, 0, 8193, 19, 0.5, 1, 90, 1				; falling sigmoid
	giExpFall	ftgen	0, 0, 8193, 5, 1, 8193, 0.00001				; exponential decay
	giTriangleWin 	ftgen	0, 0, 8193, 7, 0, 4096, 1, 4096, 0			; triangular window 

; default init (silent) ambient background tables (STATIC TABLE NUMS 91-98)
    iAmbInitSize        = 262144
    giAmbient1          ftgen	91, 0, iAmbInitSize, 2, 0	; empty buffer
    giAmbient2          ftgen	92, 0, iAmbInitSize, 2, 0	; empty buffer
    giAmbient3          ftgen	93, 0, iAmbInitSize, 2, 0	; empty buffer
    giAmbient4          ftgen	94, 0, iAmbInitSize, 2, 0	; empty buffer
    giAmbientTabNum     = 0
    ;giAmbientTabs       ftgen   0, 0, 4, -2, giAmbient1,giAmbient2,giAmbient3,giAmbient4

; panalyzer tables
	giPanalyzeTemp0		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp1		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp2		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp3		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp4		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp5		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp6		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp7		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp8		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp9		ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTemp10	ftgen	0, 0, 1024, 2, 0	; empty buffer
	giPanalyzeTempRef	ftgen	0, 0, 1024, 2, 0	; empty buffer

	gkPanalyzeArr[]		init 11

; secondary synth, keep track of tables currently in use
        giTablesInUseSecondary  ftgen   0, 0, 1024, 2, 0        ; init empty
        
