
;******************************
; audio file playback 
	instr 60
print p1,p2,p3,p5,p6,p7,p8,p9,p10,p11
	Ssound	        strget p4
        ilen            filelen Ssound
                
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator        
	aout	        diskin Ssound, ispeed
        aout            = aout * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Spectral voice 1
	instr 61

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator        
        iscale          = semitone(1)
        ishift          = -50;p13
        ilowest         = 100;p14
                
        ifftsize	= 1024
        ioverlap	= ifftsize / 4
        iwinsize	= ifftsize
        iwinshape	= 1; von-Hann window
        ibuffer, ilen, k0 FileToPvsBuf	Ssound, ifftsize, ioverlap, iwinsize, iwinshape
        p3              = ilen/ispeed
	ktmpnt		linseg		0, p3, ilen
        fread 		pvsbufread  	ktmpnt, ibuffer
        ikeepform       = 1
	fscale  	pvscale  fread,iscale, ikeepform
        ikeepform2      = 0
	fshift  	pvshift  fscale, ishift, ilowest, ikeepform2
        aout		pvsynth	fshift
        aout            butterhp aout, 400
        aout            = aout * iamp * 2 * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Spectral voice 2
	instr 62

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator          
        ktimeAmp        = 0.4  ; p12
        ktimeFreq       = 0.06 ; p13
        kblurTime       = 0.15 ; p14
        
        ifftsize	= 1024
        ioverlap	= ifftsize / 4
        iwinsize	= ifftsize
        iwinshape	= 1; von-Hann window
        ibuffer, ilen, k0 FileToPvsBuf	Ssound, ifftsize, ioverlap, iwinsize, iwinshape
        p3              = ilen/ispeed
	ktmpnt		linseg		0, p3, ilen
        fread 		pvsbufread  	ktmpnt, ibuffer
	iframetime	= (ioverlap/sr)						; time per fft frame rate 
	kamountAmp	= iframetime/ktimeAmp					; smoothing frequency (in fraction of 1/2 fft framerate)
	kamountFreq	= iframetime/ktimeFreq					; smoothing frequency (in fraction of 1/2 fft framerate)
	fsmooth		pvsmooth fread, kamountAmp, kamountFreq
	iMax		= 1.7
	fblur		pvsblur fread, kblurTime, iMax

        aout1		pvsynth	fsmooth
        aout2		pvsynth	fblur
        aout1           = aout1 * iamp * 4 
        aout2           = aout2 * iamp * 3 
        aout            = (aout1+aout2) * 0.5 * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Spectral voice 3
	instr 63

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator          
        ktimeAmp        = 0.3  
        ktimeFreq       = 0.06 
        kblurTime       = 0.2  
        kscale          = semitone(-0.3)
        kshift          = 100
        klowest         = 700

        ifftsize	= 1024
        ioverlap	= ifftsize / 4
        iwinsize	= ifftsize
        iwinshape	= 1; von-Hann window
        ibuffer, ilen, k0 FileToPvsBuf	Ssound, ifftsize, ioverlap, iwinsize, iwinshape
        p3              = ilen/ispeed
	ktmpnt		linseg		0, p3, ilen
        fread 		pvsbufread  	ktmpnt, ibuffer
	iframetime	= (ioverlap/sr)						; time per fft frame rate 
	kamountAmp	= iframetime/ktimeAmp					; smoothing frequency (in fraction of 1/2 fft framerate)
	kamountFreq	= iframetime/ktimeFreq					; smoothing frequency (in fraction of 1/2 fft framerate)
	fsmooth		pvsmooth fread, kamountAmp, kamountFreq
	iMax		= 1.7
	fblur		pvsblur fread, kblurTime, iMax
        fblurg          pvsgain fblur, 0.8
        fmix            pvsmix  fsmooth, fblurg

        ikeepform       = 0
	fscale  	pvscale  fmix,kscale, ikeepform
        ikeepform2      = 0
	fshift  	pvshift  fscale, kshift, klowest, ikeepform2

        aout		pvsynth	fshift 
        aout            butterhp aout, 400
        aout            butterhp aout, 400
        aout            butterhp aout, 400
        aout           = aout * iamp * 4 * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Spectral voice 4
	instr 64

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator          
        ifftsize	= 1024
        ioverlap	= ifftsize / 4
        iwinsize	= ifftsize
        iwinshape	= 1; von-Hann window
	inbins	        = (ifftsize/2)
        ibuffer, ilen, k0 FileToPvsBuf	Ssound, ifftsize, ioverlap, iwinsize, iwinshape
        p3              = ilen/ispeed
	ktmpnt		linseg		0, p3, ilen
        fread 		pvsbufread  	ktmpnt, ibuffer
        fnois           pvsgain fread, 1
	ifn_amp		ftgen 0,0,inbins,10,1        				; make ftable for spectral profile (amp)
	ifn_freq 	ftgen 0,0,inbins,10,1        				; make ftable for spectral profile (amp)
	kflag_in   	pvsftw fnois,ifn_amp,ifn_freq      			; export to table,
	if kflag_in == 0 kgoto contin						; only proc when frame is ready
	        kcount  = 0
process:
	        klev    = 1
	        knois   rnd31 klev, 1
	        kfq     table kcount, ifn_freq
	                tablew  kfq+(kfq*knois), kcount, ifn_freq               ; add noise, noise level dependent on value in table
                kcount  = kcount + 1
	        if kcount < inbins goto process
	; read modified data back to fsrc
	        	pvsftr fnois, ifn_amp,ifn_freq
contin:
        ktimeAmp        = 0.1
        ktimeFreq       = 0.02
	iframetime	= (ioverlap/sr)						; time per fft frame rate 
	kamountAmp	= iframetime/ktimeAmp					; smoothing frequency (in fraction of 1/2 fft framerate)
	kamountFreq	= iframetime/ktimeFreq					; smoothing frequency (in fraction of 1/2 fft framerate)
	fsmooth		pvsmooth fnois, kamountAmp, kamountFreq

        fshift          pvshift fread, 500, 1800
        fshift2         pvshift fshift, -100, 500
        aout1		pvsynth	fshift2
        aout2		pvsynth	fsmooth
        aout            = (aout1+aout2)
        aout            butterhp aout, 400
        aout            butterhp aout, 600
        aout            butterhp aout, 800
        aout            butterbr aout, 1000, 700
        aout            butterbr aout, 1000, 500
        aout           = aout * iamp * 4 * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Spectral voice 5
	instr 65

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator          
        ifftsize	= 1024
        ioverlap	= ifftsize / 4
        iwinsize	= ifftsize
        iwinshape	= 1; von-Hann window
	inbins	        = (ifftsize/2)
        ibuffer, ilen, k0 FileToPvsBuf	Ssound, ifftsize, ioverlap, iwinsize, iwinshape
        p3              = ilen/ispeed
	ktmpnt		linseg		0, p3, ilen
        fread 		pvsbufread  	ktmpnt, ibuffer
        fnois           pvsgain fread, 1
	ifn_amp		ftgen 0,0,inbins,10,1        				; make ftable for spectral profile (amp)
	ifn_freq 	ftgen 0,0,inbins,10,1        				; make ftable for spectral profile (amp)
	kflag_in   	pvsftw fnois,ifn_amp,ifn_freq      			; export to table,
	if kflag_in == 0 kgoto contin						; only proc when frame is ready
	        kcount  = 0
process:
	        klev    = 1
	        knois   rnd31 klev, 1
	        kfq     table kcount, ifn_freq
	                tablew  kfq+(kfq*knois), kcount, ifn_freq               ; add noise, noise level dependent on value in table
                kcount  = kcount + 1
	        if kcount < inbins goto process
	; read modified data back to fsrc
	        	pvsftr fnois, ifn_amp,ifn_freq
contin:
        ktimeAmp        = 0.4
        ktimeFreq       = 0.3
	iframetime	= (ioverlap/sr)						; time per fft frame rate 
	kamountAmp	= iframetime/ktimeAmp					; smoothing frequency (in fraction of 1/2 fft framerate)
	kamountFreq	= iframetime/ktimeFreq					; smoothing frequency (in fraction of 1/2 fft framerate)
	fsmooth		pvsmooth fnois, kamountAmp, kamountFreq
                
        fshift          pvshift fread, 700, 1500
        fshift2         pvshift fread, 100, 700
	iMax		= 1.7
	kblurTime       = 0.12
	fblur		pvsblur fshift2, kblurTime, iMax
        
        ashift		pvsynth	fshift
        ashift2		pvsynth	fshift2
        ablur           pvsynth fblur
        asmooth         pvsynth fsmooth
        anois           pvsynth fnois

        aout           = (ashift * 0.9) +\
                         (ashift2 * 1) +\
                         (ablur * 1) +\
                         (asmooth * 0.9)+\
                         (anois * 1.5)
        aout            butterhp aout, 300
        aout            butterhp aout, 400
        aout            butterhp aout, 600
        aout           = aout * iamp * 1.5 * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin


;******************************
; Partikkel voice 1, "slowdown but at original speed"
	instr 66

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        ichangeRate     = 2                                     ; grain subsegment spawn rate
	iTimeRate2	= ispeed*0.6				; grain subsegment time pointer rate
	idurSamplepos   = (idur/ispeed)
        p3              = (idur/ispeed)+((1/ichangeRate)*iTimeRate2)
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator         
        iSound  ftgentmp 0, 0, 0, 1, Ssound, 0,0,0
#include "partikkel3_self.inc"
        aout           = aout * iamp * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin

;******************************
; Partikkel voice 1, "PSGS", Pitch Synchronous Granular Synthesis
	instr 67

	Ssound	        strget p4
        ilen            filelen Ssound
        
; regular instr parameters        
        iamp            = ampdbfs(p7)                           ; amp
        imaxamp         = p8                                    ; max amp in this sound file
        ivoice          = p9                                    ; 1 or 2, for primary or secondary association
        idel2           = ampdbfs(p10)                           ; send to delay (2)
        irvb2           = ampdbfs(p11)                          ; send to reverb (2)
        ispeed          = p12                                   ; playback speed

; start and end not fully implemented yet, read segment info from sound/markerfile to get start and end points 
        istart          = p5                                    ; segment start
        iend            = p6                                    ; segment end
        if iend == 0 then
        iend            = ilen
        endif
        idur            = iend - istart
        p3              = idur/ispeed
        aenv            linen   iamp, 0.01, p3, 0.2

; sound generator         
        iSound          ftgentmp 0, 0, 0, 1, Ssound, 0,0,0
        istrlen         strlen Ssound
        Spitchfile      strsub Ssound, 0, istrlen-4
        Spitchfile      strcat Spitchfile, "_pitch.txt"
        iPitchFn        ftgentmp 0, 0, 1048576, -23, Spitchfile
        ;ftload 		Spitchfile, 1, iPitchFn
        ;puts Spitchfile, 1
        iformant	= semitone(1)		; formant change in semitones
        itransp		= semitone(-12)		; transposition in semitones
#include "partikkel4_psgs.inc"
        aout           = aout * iamp * aenv * (1/imaxamp)

; output
        if ivoice == 1 then
  	  iout1         = 1
  	  iout2         = 0
  	elseif ivoice == 2 then
  	  iout1         = 0
  	  iout2         = 1
  	endif
		        chnmix aout*iout1, "MasterOut1"
		        chnmix aout*iout2, "SecondaryDry"
                        chnmix aout*idel2,  "SecondaryDelaySend"
                        chnmix aout*irvb2,  "SecondaryReverbSend"

	endin
