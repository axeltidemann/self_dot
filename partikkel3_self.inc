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


;*******************************
; setup of source waveforms
; (needs to be done first, because grain pitch and time pointer depends on source waveform lengths)
;*******************************

; source waveform selection automation
        iwaveform1      = iSound
        iwaveform2      = iSound
        iwaveform3      = iSound
        iwaveform4      = iSound
        
; get source waveform length (used when calculating transposition and time pointer)
        ifilen1         tableng  iwaveform1             ; get length of the first source waveform
        ifilen2         tableng  iwaveform2             ; same as above, for source waveform 2
        ifilen3         tableng  iwaveform3             ; same as above, for source waveform 3
        ifilen4         tableng  iwaveform4             ; same as above, for source waveform 4
        ifildur1        = ifilen1 / sr                  ; length in seconds, for the first source waveform
        ifildur2        = ifilen2 / sr                  ; same as above, for source waveform 2
        ifildur3        = ifilen3 / sr                  ; same as above, for source waveform 3
        ifildur4        = ifilen4 / sr                  ; same as above, for source waveform 4

; original pitch for each waveform, use if they should be transposed individually
        kwavekey1       = 1             ; individual source pitch
        kwavekey2       = 1.01          ; individual source pitch
        kwavekey3       = 0.99          ; individual source pitch
        kwavekey4       = 1.02          ; individual source pitch

; set original key dependant on waveform length 
        kwavekey1       divz kwavekey1, ifildur1, 1
        kwavekey2       divz kwavekey2, ifildur2, 1
        kwavekey3       divz kwavekey3, ifildur3, 1
        kwavekey4       divz kwavekey4, ifildur4, 1
    
        isposEnd        = divz(idur,ifildur1, 1) ; section of table used, (normalized) 
        asamplepos0     line 0, idurSamplepos, isposEnd
        ksamplepos0     downsamp asamplepos0
        asamplepos1     init 0
        kmetro          metro ichangeRate
        kcount          init -1
        kcount          += kmetro
        kcount          = kcount%4
        ;printk2 kcount
        kcpsamp1        init 0
        kcpsamp2        init 0
        kcpsamp3        init 0
        kcpsamp4        init 0
        if kmetro > 0 then
          if kcount == 0 then
            kcpsamp1    = ichangeRate*0.25
            reinit spos1
          elseif kcount == 1 then
            kcpsamp2    = ichangeRate*0.25
            reinit spos2
          elseif kcount == 2 then
            kcpsamp3    = ichangeRate*0.25
            reinit spos3
          elseif kcount == 3 then
            kcpsamp4    = ichangeRate*0.25
            reinit spos4
          endif
        endif

spos1:
        iduramp1        divz 1, i(kcpsamp1), 1/kr
        ks1amp          oscil1, 0, 1, iduramp1, giHanning
        aspos1          phasor divz(iTimeRate2, ifildur1, 1)
        ipos1           = i(ksamplepos0)
        asamplepos1     = ipos1 + aspos1
        rireturn
spos2:
        iduramp2        divz 1, i(kcpsamp2), 1/kr
        ks2amp          oscil1, 0, 1, iduramp2, giHanning
        aspos2          phasor divz(iTimeRate2, ifildur1, 1)
        ipos2           = i(ksamplepos0)
        asamplepos2     = ipos2 + aspos2
        rireturn
spos3:
        iduramp3        divz 1, i(kcpsamp3), 1/kr
        ks3amp          oscil1, 0, 1, iduramp3, giHanning
        aspos3          phasor divz(iTimeRate2, ifildur1, 1)
        ipos3           = i(ksamplepos0)
        asamplepos3     = ipos3 + aspos3
        rireturn
spos4:
        iduramp4        divz 1, i(kcpsamp4), 1/kr
        ks4amp          oscil1, 0, 1, iduramp4, giHanning
        aspos4          phasor divz(iTimeRate2, ifildur1, 1)
        ipos4           = i(ksamplepos0)
        asamplepos4     = ipos4 + aspos4
        rireturn
donespos:

; samplepos rDev
        ksampRdevAmt    = divz(0.001, ifildur1, 0.0001)
        ksampRdev1      randh ksampRdevAmt, 30
        ksampRdev2      randh ksampRdevAmt, 30
        ksampRdev3      randh ksampRdevAmt, 30
        ksampRdev4      randh ksampRdevAmt, 30
        asamplepos1     = asamplepos1+ksampRdev1
        asamplepos2     = asamplepos2+ksampRdev2
        asamplepos3     = asamplepos3+ksampRdev3
        asamplepos4     = asamplepos4+ksampRdev4

        ; wave mix masking. 
        ; Set gain per source waveform per grain, 
        ; in groups of 5 amp values, reflecting source1, source2, source3, source4, and the 5th slot is for trainlet amplitude.
        iwaveamptab     ftgentmp        0, 0, 32, -2, 0, 0,   0,0,0,0,0
        ks1amp          init 0
        ks2amp          init 0
        ks3amp          init 0
        ks4amp          init 0
        
                        tablew ks1amp, 2, iwaveamptab
                        tablew ks2amp, 3, iwaveamptab
                        tablew ks3amp, 4, iwaveamptab
                        tablew ks4amp, 5, iwaveamptab


;*******************************
; other granular synthesis parameters
;*******************************

; amplitude
        kamp            = ampdbfs(-6)                           ; output amplitude

; grain rate
        kGrainRate      = 120                                   ; number of grains per second

; sync 
        kdevAmount      = 0.3                                   ; synchronous/asynchronous balance
        async           probabilityClock kGrainRate             ; asynchronous clock
        async           = async*kdevAmount                      ; Balance the weight of the sync pulse
        agrainrate      = kGrainRate*(1-kdevAmount)             ; ... with the rate of the internal clock

; grain rate FM
        kGrFmFreq       = 10                                    ; FM freq for modulating the grainrate 
        kGrFmIndex      = 0.0                                   ; FM index for modulating the grainrate (normally kept in a 0.0 to 1.0 range)
        iGrFmWave       = giSine                                ; FM waveform, for modulating the grainrate 
        aGrFmSig        oscil kGrFmIndex, kGrFmFreq, iGrFmWave  ; audio signal for frequency modulation of grain rate
        agrainrate      = kGrainRate + (aGrFmSig*kGrainRate)    ; add the modulator signal to the grain rate signal

; distribution 
        kdistribution   = 0.3                                           ; grain random distribution in time
        idisttab        ftgentmp        0, 0, 16, 16, 1, 16, -10, 0     ; probability distribution for random grain masking

; grain shape
        kGrainDur       = 4                                         ; length of each grain relative to grain rate 
        kduration       = divz((kGrainDur*1000), kGrainRate, 100)    ; grain dur in milliseconds, relative to grain rate

        ienv_attack     = giSigmoRise                           ; grain attack shape (from table)
        ienv_decay      = giSigmoFall                           ; grain decay shape (from table)
        ksustain_amount = 0                                     ; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
        ka_d_ratio      = 0.5                                   ; balance between attack time and decay time, 0.0 = zero attack time and full decay time
        kenv2amt        = 0                                     ; amount of secondary enveloping per grain (e.g. for fof synthesis)
        ienv2tab        = giExpFall                             ; secondary grain shape (from table), enveloping the whole grain if used

; grain pitch (transpose, or "playback speed")
        kwavfreq        = 1                                     ; transposition factor (playback speed) of audio inside grains, 

; pitch sweep
        ksweepshape             = 0.75
        iwavfreqstarttab        ftgentmp        0, 0, 16, -2, 0, 0,   1         ; start freq scalers, per grain
        iwavfreqendtab          ftgentmp        0, 0, 16, -2, 0, 0,   1         ; end freq scalers, per grain

; FM of grain pitch (playback speed)
        kPtchFmFreq     = 440                                                   ; FM freq, modulating waveform pitch
        kPtchFmIndex    = 0                                                     ; FM index, modulating waveform pitch
        iPtchFmWave     = giSine                                                ; FM waveform, modulating waveform pitch
        ifmamptab       ftgentmp        0, 0, 16, -2, 0, 0,   1                 ; FM index scalers, per grain
        ifmenv          = giTriangleWin                                         ; FM index envelope, over each grain (from table)
        kPtchFmIndex    = kPtchFmIndex + (kPtchFmIndex*kPtchFmFreq*0.00001)     ; FM index scaling formula
        awavfm          oscil   kPtchFmIndex, kPtchFmFreq, iPtchFmWave          ; Modulator signal for frequency modulation inside grain

; trainlet parameters
        icosine         = giCosine                              ; needs to be a cosine wave to create trainlets
        kTrainCps       = kGrainRate/kGrainDur                  ; create a single cycle of a trainlet inside each grain
        knumpartials    = 4                                     ; number of partials in trainlet
        kchroma         = 2                                     ; chroma, falloff of partial amplitude towards sr/2

; masking
        ; gain masking table, amplitude for individual grains
        igainmasks      ftgentmp        0, 0, 16, -2, 0, 1,   1, 1

        kgainOct        = 0                                     ; octaviation, amp mod on every 2nd grain
                        tablew  1-sqrt(kgainOct), 3, igainmasks ; write modified gain mask, every 2nd grain will get a modified amplitude 


        ; channel masking table, output routing for individual grains (zero based, a value of 0.0 routes to output 1)
        ichannelmasks   ftgentmp        0, 0, 16, -2,  0, 3,  0.5, 0.5, 0.5, 0.5
        
        ; random masking (muting) of individual grains
        krandommask     = 0                                     ; stochastic masking, masking off for the first 7 seconds, then increase stochastic masking

; system parameter
        imax_grains     = 100                           ; max number of grains per k-period

;***********


a1,a2,a3,a4,a5,a6,a7,a8 partikkel agrainrate, kdistribution, idisttab, async, kenv2amt, ienv2tab, \
                  ienv_attack, ienv_decay, ksustain_amount, ka_d_ratio, kduration, kamp, igainmasks, \
                  kwavfreq, ksweepshape, iwavfreqstarttab, iwavfreqendtab, awavfm, \
                  ifmamptab, ifmenv, icosine, kTrainCps, knumpartials, \
                  kchroma, ichannelmasks, krandommask, iwaveform1, iwaveform2, iwaveform3, iwaveform4, \
                  iwaveamptab, asamplepos1, asamplepos2, asamplepos3, asamplepos4, \
                  kwavekey1, kwavekey2, kwavekey3, kwavekey4, imax_grains

; out
        aout            = a1*1.3
        aout            butterhp aout, 150


