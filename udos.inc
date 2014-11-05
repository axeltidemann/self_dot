;***************************************************
;user defined opcode, asynchronous clock
			opcode		probabilityClock, a, k
	kdens		xin
			setksmps 1
	krand		rnd31	1, 1
	krand		= (krand*0.5)+0.5
	ktrig		= (krand < kdens/kr ? 1 : 0)
	atrig		upsamp ktrig
			xout atrig
			endop

;***************************************************
; Transient detection udo

	opcode TransientDetect, kk,kikkkk
	kin, iresponse, ktthresh, klowThresh, kdecThresh, kdoubleLimit xin 
/*
	iresponse	= 10 		; response time in milliseconds
	ktthresh	= 6		; transient trig threshold 
	klowThresh	= -60		; lower threshold for transient detection
	kdoubleLimit	= 0.02		; minimum duration between events, (double trig limit)
	kdecThresh	= 6		; retrig threshold, how much must the level decay from its local max before allowing new transient trig
*/	
	kinDel		delayk	kin, iresponse/1000		; delay with response time for comparision of levels
	ktrig		= ((kin > kinDel + ktthresh) ? 1 : 0) 	; if current rms plus threshold is larger than previous rms, set trig signal to current rms
	klowGate	= (kin < klowThresh? 0 : 1)		; gate to remove transient of low level signals
	ktrig		= ktrig * klowGate			; activate gate on trig signal
	ktransLev	init 0
	ktransLev	samphold kin, 1-ktrig			; read amplitude at transient
	
	kreGate		init 1					; retrigger gate, to limit transient double trig before signal has decayed (decThresh) from its local max
	ktrig		= ktrig*kreGate				; activate gate
	kmaxAmp		init -99999
	kmaxAmp		max kmaxAmp, kin			; find local max amp
	kdiff		= kmaxAmp-kin				; how much the signal has decayed since its local max value
	kreGate		limit kreGate-ktrig, 0, 1		; mute when trig detected
	kreGate		= (kdiff > kdecThresh ? 1 : kreGate)	; re-enable gate when signal has decayed sufficiently
	kmaxAmp		= (kreGate == 1 ? -99999 : kmaxAmp)	; reset max amp gauge

	; avoid closely spaced transient triggers (first trig priority)
	kdouble		init 1
	ktrig		= ktrig*kdouble
	if ktrig > 0 then
	reinit double
	endif
	double:
    idoubleLimit  = i(kdoubleLimit)	
    idoubleLimit    limit idoubleLimit, 0.1, 5
    print idoubleLimit
    kdouble		linseg	0, idoubleLimit, 0, 0, 1, 1, 1
	rireturn
    printk2 kdouble, 30

	xout ktrig, kdiff
	endop

;***************************************************
opcode FileToPvsBuf, iik, Siiiiii
        ;; by joachim heintz 2009, modified by Oeyvind Brandtsegg 2014
        ;;writes an audio file at the first k-cycle to a fft-buffer (via pvsbuffer)

        Sfile, ifftsize, ioverlap, iwinsize, iwinshape, istart, iend xin

        ktimek		    timeinstk
if ktimek == 1 then
;        ilen		    filelen	Sfile
        kcycles	        = (iend-istart) * kr; number of k-cycles to write the fft-buffer
        kcount		    init 0
        ;print istart, iend
        ;1.712 
        ;2.638 

loop:
        ain		        diskin	Sfile, 1, istart
        fftin		    pvsanal	ain, ifftsize, ioverlap, iwinsize, iwinshape
        ibuf, ktim	    pvsbuffer fftin, (iend-istart) + (ifftsize / sr)
	                    loop_lt	kcount, 1, kcycles, loop
		                xout ibuf, (iend-istart), ktim
endif
endop
