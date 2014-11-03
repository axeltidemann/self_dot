;******************************************************
; ambient background partikkel instr
;******************************************************

; select source waveforms 
        kwaveform1      table 0, giAmbientTabs  ; source audio waveform 1
        kwaveform2      table 0, giAmbientTabs  ; ...
        kwaveform3      table 0, giAmbientTabs
        kwaveform4      table 0, giAmbientTabs

; get source waveform length (used when calculating transposition and time pointer)
        kfilen1         tableng  kwaveform1             ; get length of the first source waveform
        kfilen2         tableng  kwaveform2             ; same as above, for source waveform 2
        kfilen3         tableng  kwaveform3             ; same as above, for source waveform 3
        kfilen4         tableng  kwaveform4             ; same as above, for source waveform 4
        kfildur1        = kfilen1 / sr                  ; length in seconds, for the first source waveform
        kfildur2        = kfilen2 / sr                  ; same as above, for source waveform 2
        kfildur3        = kfilen3 / sr                  ; same as above, for source waveform 3
        kfildur4        = kfilen4 / sr                  ; same as above, for source waveform 4

; original pitch for each waveform
        kwavekey1       = 1
        kwavekey2       = 1
        kwavekey3       = 1
        kwavekey4       = 1

; set original key dependant on waveform length 
        kwavekey1       = kwavekey1/kfildur1
        kwavekey2       = kwavekey2/kfildur2
        kwavekey3       = kwavekey3/kfildur3
        kwavekey4       = kwavekey4/kfildur4

; time pointer (phase). This can be independent for each source waveform.

        kTimeRate       = 1                             ; time pointer rate
        asamplepos1     phasor kTimeRate / kfildur1     ; phasor from 0 to 1, scaled to the length of the first source waveform
        asamplepos2     phasor kTimeRate / kfildur2     ; same as above, scaled for source wave 2
        asamplepos3     phasor kTimeRate / kfildur3     ; same as above, scaled for source wave 3
        asamplepos4     phasor kTimeRate / kfildur4     ; same as above, scaled for source wave 4

;*******************************
; other granular synthesis parameters
;*******************************

; amplitude
        ivoice          = 1
        kamp            = ampdbfs(-1)                          ; output amplitude
        
; sync
        async           = 0.0                                   ; set the sync input to zero (disable external sync)
        Syncin          sprintf "sync%i", ivoice
        Syncout          sprintf "sync%i", ivoice+1
        async           chnget Syncin                           ; sync input 
        ksyncAmt        = 0;linseg 1, 2, 0, 1, 0
        async           = async*ksyncAmt

; grain rate
        kgrainrate      = 12.0                                  ; number of grains per second
        kgrainrate      rspline 0.5, 15, 0.3, 2
        krateDev        randh 1, kgrainrate
        krateDev        pow krateDev, 2
        kgrainrate      = kgrainrate+(krateDev*kgrainrate)

; grain rate FM
        kGrFmFreq       = kgrainrate/4                          ; FM freq for modulating the grainrate 
        kGrFmIndex      = 0.0                                   ; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
        iGrFmWave       = giSine                                ; FM waveform, for modulating the grainrate 
        aGrFmSig        oscil kGrFmIndex, kGrFmFreq, iGrFmWave  ; audio signal for frequency modulation of grain rate
        agrainrate      = kgrainrate + (aGrFmSig*kgrainrate)    ; add the modulator signal to the grain rate signal

; distribution 
        kdistribution   = 0.0                                           ; grain random distribution in time
        idisttab        ftgenonce       0, 0, 16, 16, 1, 16, -10, 0     ; probability distribution for random grain masking

; grain shape
        kGrainDur       = 0.1                                   ; length of each grain relative to grain rate 
        kRelDur         = (kGrainDur*1000)/kgrainrate           ; grain dur in milliseconds, relative to grain rate
        kduration       = 20+kRelDur

        ienv_attack     = giSigmoRise                           ; grain attack shape (from table)
        ienv_decay      = giSigmoFall                           ; grain decay shape (from table)
        ksustain_amount = 0.0                                   ; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
        ka_d_ratio      = 0.1                                   ; balance between attack time and decay time, 0.0 = zero attack time and full decay time

        kenv2amt        = 0.0                                   ; amount of secondary enveloping per grain (e.g. for fof synthesis)
        ienv2tab        = giExpFall                             ; secondary grain shape (from table), enveloping the whole grain if used

; grain pitch (transpose, or "playback speed")
        ;kwavfreq       = 1                                     ; transposition factor (playback speed) of audio inside grains, 
        kwavDev      rspline -1, 1, 0.3, 2
        kwavfreq        = 1*semitone(kwavDev)

; pitch sweep
        ksweepshape             = 0.5                                           ; grain wave pitch sweep shape (sweep speed), 0.5 is linear sweep
        iwavfreqstarttab        ftgenonce       0, 0, 16, -2, 0, 0,   1         ; start freq scalers, per grain
        iwavfreqendtab          ftgenonce       0, 0, 16, -2, 0, 0,   1         ; end freq scalers, per grain

; FM of grain pitch (playback speed)
        kPtchFmFreq     = 440                                                   ; FM freq, modulating waveform pitch
        kPtchFmIndex    = 0                                                     ; FM index, modulating waveform pitch
        iPtchFmWave     = giSine                                                ; FM waveform, modulating waveform pitch
        ifmamptab       ftgenonce       0, 0, 16, -2, 0, 0,   1                 ; FM index scalers, per grain
        ifmenv          = giTriangleWin                                         ; FM index envelope, over each grain (from table)
        kPtchFmIndex    = kPtchFmIndex + (kPtchFmIndex*kPtchFmFreq*0.00001)     ; FM index scaling formula
        awavfm          oscil   kPtchFmIndex, kPtchFmFreq, iPtchFmWave          ; Modulator signal for frequency modulation inside grain

; trainlet parameters
        icosine         = giCosine                              ; needs to be a cosine wave to create trainlets
        kTrainCps       = kgrainrate                            ; set cps equal to grain freq, creating a single cycle of a trainlet inside each grain
        knumpartials    = 7                                     ; number of partials in trainlet
        kchroma         = 3                                     ; chroma, falloff of partial amplitude towards sr/2

; masking
        igainmasks      ftgenonce       0, 0, 16, -2, 0, 0,   1
        igainmasks2     ftgenonce       0, 0, 16, -2, 0, 1,   1, 0
        ichannelmasks   ftgenonce       0, 0, 16, -2,  0, 4,  0, 1, 0, 0.5, 1
        ichannelmasks2  ftgenonce       0, 0, 16, -2,  0, 6,  0, 0, 1, 0, 1, 0, 1
        
        ; random masking (muting) of individual grains
        krandommask     = 0

        ; wave mix masking. 
        ; Set gain per source waveform per grain, 
        ; in groups of 5 amp values, reflecting source1, source2, source3, source4, and the 5th slot is for trainlet amplitude.
        iwaveamptab     ftgenonce       0, 0, 32, -2, 0, 0,   1,1,1,1,0

; system parameter
        imax_grains     = 100                           ; max number of grains per k-period

      
a1,a2                   partikkel \                                             ;                                       
                        kgrainrate, \                                           ; grains per second                     
                        kdistribution, idisttab, async, \                       ; synchronous/asynchronous              
                        kenv2amt, ienv2tab, ienv_attack, ienv_decay, \          ; grain envelope (advanced)             
                        ksustain_amount, ka_d_ratio, kduration, \               ; grain envelope                        
                        kamp, \                                                 ; amp                                   
                        igainmasks, \                                           ; gain masks (advanced)                 
                        kwavfreq, \                                             ; grain pitch (playback frequency)      
                        ksweepshape, iwavfreqstarttab, iwavfreqendtab, \        ; grain pith sweeps (advanced)          
                        awavfm, ifmamptab, ifmenv, \                            ; grain pitch FM (advanced)             
                        icosine, kTrainCps, knumpartials, kchroma, \            ; trainlets                             
                        ichannelmasks, \                                        ; channel mask (advanced)
                        krandommask, \                                          ; random masking of single grains       
                        kwaveform1, kwaveform2, kwaveform3, kwaveform4, \       ; set source waveforms                  
                        iwaveamptab, \                                          ; mix source waveforms
                        asamplepos1, asamplepos2, asamplepos3, asamplepos4, \   ; read position for source waves        
                        kwavekey1, kwavekey2, kwavekey3, kwavekey4, \           ; individual transpose for each source
                        imax_grains, ivoice                                     ; system parameter (advanced)

;async1,aphase           partikkelsync, ivoice 
;                        chnset async1, Syncout


        ivoice2         = ivoice+1
        kamp2           = ampdbfs(-10)                          ; output amplitude
        ;kgrainrate2     = 200
        kgrainrate2     rspline 0.4, 250, 0.1, 2
        krateDev2       randh 1, kgrainrate2
        krateDev2       pow krateDev2, 2
        ;kgrainrate2     = kgrainrate2+(krateDev2*kgrainrate2)
        kGrainDur2      = 3                                     ; length of each grain relative to grain rate 
        kduration2      = (kGrainDur2*1000)/kgrainrate2         ; grain dur in milliseconds, relative to grain rate
        kwavDev2        rspline -1, 1, 0.3, 2
        kwavfreq2       = 1*semitone(kwavDev2)
        ka_d_ratio2     = 0.3                                   ; balance between attack time and decay time, 0.0 = zero attack time and full decay time
  
        kTimeRate2      = 0.5                           ; time pointer rate
        asamplepos21    phasor kTimeRate2 / kfildur1    ; phasor from 0 to 1, scaled to the length of the first source waveform
        asamplepos22    phasor kTimeRate2 / kfildur2    ; same as above, scaled for source wave 2
        asamplepos23    phasor kTimeRate2 / kfildur3    ; same as above, scaled for source wave 3
        asamplepos24    phasor kTimeRate2 / kfildur4    ; same as above, scaled for source wave 4

        kposDevAmt2     = 0.4
        kposDev21       randh kposDevAmt2, kgrainrate2
        kposDev22       randh kposDevAmt2, kgrainrate2
        kposDev23       randh kposDevAmt2, kgrainrate2
        kposDev24       randh kposDevAmt2, kgrainrate2
        ; mix initial phase and moving phase value (moving phase only for sampled waveforms, single cycle waveforms use static samplepos)
        asamplepos21    = asamplepos21+kposDev21
        asamplepos22    = asamplepos22+kposDev22
        asamplepos23    = asamplepos23+kposDev23
        asamplepos24    = asamplepos24+kposDev24

a21,a22                 partikkel \                                             ;                                       
                        kgrainrate2, \                                          ; grains per second                     
                        kdistribution, idisttab, async, \                       ; synchronous/asynchronous              
                        kenv2amt, ienv2tab, ienv_attack, ienv_decay, \          ; grain envelope (advanced)             
                        ksustain_amount, ka_d_ratio2, kduration2, \             ; grain envelope                        
                        kamp2, \                                                        ; amp                                   
                        igainmasks, \                                           ; gain masks (advanced)                 
                        kwavfreq2, \                                            ; grain pitch (playback frequency)      
                        ksweepshape, iwavfreqstarttab, iwavfreqendtab, \        ; grain pith sweeps (advanced)          
                        awavfm, ifmamptab, ifmenv, \                            ; grain pitch FM (advanced)             
                        icosine, kTrainCps, knumpartials, kchroma, \            ; trainlets                             
                        ichannelmasks, \                                        ; channel mask (advanced)
                        krandommask, \                                          ; random masking of single grains       
                        kwaveform1, kwaveform2, kwaveform3, kwaveform4, \       ; set source waveforms                  
                        iwaveamptab, \                                          ; mix source waveforms
                        asamplepos21, asamplepos22, asamplepos23, asamplepos24, \       ; read position for source waves        
                        kwavekey1, kwavekey2, kwavekey3, kwavekey4, \           ; individual transpose for each source
                        imax_grains, ivoice                                     ; system parameter (advanced)

a1 = a1+a21
a2 = a2+a22

