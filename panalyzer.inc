
	aLeft	chnget "in1"
	aRight	chnget "in2"
	aRight = aRight * ampdbfs(1)

; preprocess
/*
    aLeft           butterhp aLeft, 400
    aRight          butterhp aRight, 400
    aLeft           butterhp aLeft, 300
    aRight          butterhp aRight, 300
*/
; amplitude comparision
	krmsLg		    rms aLeft
	krmsRg		    rms aRight
	armsL           follow2 aLeft, 0.01, 0.5
	armR            follow2 aRight, 0.01, 0.5
	krmsL           downsamp armsL
	krmsR           downsamp armR
	ktempdiv	    divz (krmsL - krmsR), (krmsL+krmsR), 1
	krmsPos		    = ((ktempdiv)+1)*0.5
	krmsPosLarge	= ((ktempdiv)*3)

; gate /samphold
	kgate		    = -22;chnget "inputNoisefloor"
	kgateL		    = (dbfsamp(krmsLg) > kgate ? 1 : 0)
	kgateR		    = (dbfsamp(krmsRg) > kgate ? 1 : 0)
    ;krmsPos		    samphold krmsPos, (kgateL+kgateR)
;printk2 krmsPos
    ktransient      chnget "transient"
noDoubleTrig:
    idoublePeriod   = 1
    kdoubleMute     linseg 0, idoublePeriod, 0, 0, 1, 1, 1
    rireturn
    ktransient2     = ktransient*kdoubleMute
    kcenter         = 0.5
    if ktransient2 > 0 then
                    chnset krmsPos, "panalyzer_pan"
                    reinit noDoubleTrig
    else
                    chnset kcenter, "panalyzer_pan"
    endif
    ;ktest   chnget "panalyzer_pan"
    ;printk2 ktest, 30

/*
; below here is for backup only

; amplitude comparision
	krmsLg		rms aLeft
	krmsRg		rms aRight
	armsL           follow2 aLeft, 0.01, 0.5
	armR            follow2 aRight, 0.01, 0.5
	krmsL           downsamp armsL
	krmsR           downsamp armR
	krmsL           limit krmsL, 0.0001, 1
	krmsR           limit krmsR, 0.0001, 1
	krmsPos		= (((krmsL - krmsR)/(krmsL+krmsR))+1)*0.5
	;krmsPosLarge	= (((krmsL - krmsR)/(krmsL+krmsR))*3)


; itd comparision
        aLeftF          butterlp aLeft*40, 1500
        aRightF         butterlp aRight*40, 1500
;        aLeftF          butterbp aLeft*40, 700, 700
;        aRightF         butterbp aRight*40, 700, 700
;        aLeftF          lpf18 aLeft*40, 1400, 0.1, 0
;        aRightF         lpf18 aRight*40, 1400, 0.1, 0

	iMaxIDT		= (0.6/1000)*2		; max interaural time delay (approx 20cm distance, 340m per sec)
	adel0		= aLeftF
	adel1		delay aLeftF, iMaxIDT*0.1
	adel2		delay aLeftF, iMaxIDT*0.2
	adel3		delay aLeftF, iMaxIDT*0.3
	adel4		delay aLeftF, iMaxIDT*0.4
	adel5		delay aLeftF, iMaxIDT*0.5
	adel6		delay aLeftF, iMaxIDT*0.6
	adel7		delay aLeftF, iMaxIDT*0.7
	adel8		delay aLeftF, iMaxIDT*0.8
	adel9		delay aLeftF, iMaxIDT*0.9
	adel10		delay aLeftF, iMaxIDT*1.0
	aref		delay aRightF, iMaxIDT*0.5

	kr0		init 0
	kr1		init 0
	kr2		init 0
	kr3		init 0
	kr4		init 0
	kr5		init 0
	kr6		init 0
	kr7		init 0
	kr8		init 0
	kr9		init 0
	kr10		init 0

	ktrig		metro 10
	if ktrig > 0 then
	reinit starts
	endif
starts:
	kstart0		init 0
	kstart1		init 0
	kstart2		init 0
	kstart3		init 0
	kstart4		init 0
	kstart5		init 0
	kstart6		init 0
	kstart7		init 0
	kstart8		init 0
	kstart9		init 0
	kstart10	init 0
	kstartRef	init 0
	rireturn

	kstart0		tablewa giPanalyzeTemp0, adel0, 0
	kstart1		tablewa giPanalyzeTemp1, adel1, 0
	kstart2		tablewa giPanalyzeTemp2, adel2, 0
	kstart3		tablewa giPanalyzeTemp3, adel3, 0
	kstart4		tablewa giPanalyzeTemp4, adel4, 0
	kstart5		tablewa giPanalyzeTemp5, adel5, 0
	kstart6		tablewa giPanalyzeTemp6, adel6, 0
	kstart7		tablewa giPanalyzeTemp7, adel7, 0
	kstart8		tablewa giPanalyzeTemp8, adel8, 0
	kstart9		tablewa giPanalyzeTemp9, adel9, 0
	kstart10	tablewa giPanalyzeTemp10, adel10, 0
	kstartRef	tablewa giPanalyzeTempRef, aref, 0
	
	kcopyflag	= (kstart0 > 1024-ksmps ? 1 : 0)
	if kcopyflag > 0 then
			vmultv giPanalyzeTemp0, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp1, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp2, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp3, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp4, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp5, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp6, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp7, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp8, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp9, giPanalyzeTempRef, 1024
			vmultv giPanalyzeTemp10, giPanalyzeTempRef, 1024
	kr0		tabsum giPanalyzeTemp0
	kr1		tabsum giPanalyzeTemp1
	kr2		tabsum giPanalyzeTemp2
	kr3		tabsum giPanalyzeTemp3
	kr4		tabsum giPanalyzeTemp4
	kr5		tabsum giPanalyzeTemp5
	kr6		tabsum giPanalyzeTemp6
	kr7		tabsum giPanalyzeTemp7
	kr8		tabsum giPanalyzeTemp8
	kr9		tabsum giPanalyzeTemp9
	kr10		tabsum giPanalyzeTemp10

	gkPanalyzeArr[0]	= kr0
	gkPanalyzeArr[1]	= kr1
	gkPanalyzeArr[2]	= kr2
	gkPanalyzeArr[3]	= kr3
	gkPanalyzeArr[4]	= kr4
	gkPanalyzeArr[5]	= kr5
	gkPanalyzeArr[6]	= kr6
	gkPanalyzeArr[7]	= kr7
	gkPanalyzeArr[8]	= kr8
	gkPanalyzeArr[9]	= kr9
	gkPanalyzeArr[10]	= kr10
	kmax, kindex	maxarray gkPanalyzeArr
	endif

; gate /samphold
	kgateL		= (dbfsamp(krmsLg) > -60 ? 1 : 0)
	kgateR		= (dbfsamp(krmsRg) > -60 ? 1 : 0)
	;krmsPos		samphold krmsPos, (kgateL+kgateR)
	;krmsPosLarge	samphold krmsPosLarge, (kgateL+kgateR)
        ;kindex          samphold kindex, (kgateL+kgateR)
	;Span		sprintfk "panning estimate: %f (delay), %f (amp)", kindex/10, krmsPos
	;		puts Span, (kcopyflag+1)*(kgateL+kgateR)

	kindex2         = kindex/10
        kindex2         mediank kindex2, 1900, 1900
        kindex2         tonek kindex2, 4

; write to chn
                        chnset kindex2, "inputPosition_idt"
                        chnset krmsPos, "inputPosition_rms"

; test write to file
	ifilewrite	= 0
	if ifilewrite > 0 then
	aindex		upsamp kindex2
;	aindex          butterlp aindex, 1
	armsPos		upsamp krmsPos
			fout "panalyze_test.wav", 14, aLeft*7, aRight*7, aindex, armsPos
        endif
*/

