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


; ******************************
; delay for secondary assoc 
        instr 77
        a1              chnget "SecondaryDelaySend"
        kTime           chnget "SecondaryDelayTime"
        kFeed           chnget "SecondaryDelayFeed"
        kFiltFq         chnget "SecondaryDelayFiltFq"
        kDryLevel       chnget "SecondaryDryLevel"
        kRvbSend        chnget "SecondaryReverbSendLevel"        
        imaxdel         = 2
        adelTim         upsamp      kTime                           ; smoothing
        adelTim         tone        adelTim, 2                      ; smoothing
        adummy          delayr imaxdel                              ; establish delay line
        aout            deltapi adelTim                             ; tap delay 
        adelw           = a1 + (aout*kFeed)                 ; mix input and feedback
        adelw           butterlp adelw, kFiltFq                     ; filter delay signal
        adelw           butterhp adelw, 100             ; low cut on feedback
                        delayw  adelw                               ; write source to delay line
        aout            butterhp aout*2, 120
        aReverbSend     = aout * ampdbfs(kRvbSend)
        aDry            = aout * ampdbfs(kDryLevel)
                        chnmix aDry, "SecondaryDry"
                        chnmix aReverbSend, "SecondaryReverbSend"
        endin

; ******************************
; reverb for secondary assoc 
        instr 78
        a1              chnget "SecondaryReverbSend"
        kFeed           chnget "SecondaryReverbFeed"
        kFiltFq         chnget "SecondaryReverbFiltFq"
        iPitchMod       = 0.9   
        aL,aR           reverbsc a1, a1, kFeed, kFiltFq, sr, iPitchMod
        aout            butterhp aL+aR, 120
                        chnmix aout, "SecondaryDry"
        endin

; ******************************
; mixer for secondary assoc 
        instr 79
; moved to master out
        endin
