; ******************************
; straight playback of audio segments
        instr 71
; FIXME
        endin

; ******************************
; partikkel instr 
	instr 72
#include "partikkel2_chn.inc"
#include "partikkel2_self.inc"
			chnmix aDry, "SecondaryDry"
			chnmix aDelaySend, "SecondaryDelaySend"
			chnmix aReverbSend, "SecondaryReverbSend"
	endin

; ******************************
; spectral synth playback of audio segments
        instr 73
; FIXME
        endin

; ******************************
; delay for secondary assoc 
	instr 77
	a1              chnget "SecondaryDelaySend"
        kTime		chnget "SecondaryDelayFeed"
        kFeed		chnget "SecondaryDelayFeed"
	kFiltFq		chnget "SecondaryDelayFiltFq"
        imaxdel         = 2
	adelTim		upsamp 	kTime				; smoothing
	adelTim		tone	adelTim, 2			; smoothing
	adummy		delayr imaxdel				; establish delay line
	aout		deltapi adelTim				; tap delay 
	adelw		= a1 + (aout*kFeed)			; mix input and feedback
	adelw		butterlp adelw, kFiltFq			; filter delay signal
			delayw	adelw				; write source to delay line
	aout		butterhp aout, 120
			chnmix aout, "SecondaryDry"
	endin

; ******************************
; reverb for secondary assoc 
	instr 78
	a1              chnget "SecondaryReverbSend"
        kFeed		chnget "SecondaryReverbFeed"
	kFiltFq		chnget "SecondaryReverbFiltFq"
	iPitchMod	= 0.9	
	aL,aR		reverbsc a1, a1, kFeed, kFiltFq, sr, iPitchMod
	aout		butterhp aL+aR, 120
			chnmix aout, "SecondaryDry"
	endin

; ******************************
; mixer for secondary assoc 
	instr 79
	klevel          chnget "SecondaryMasterLevel"
        klevel          = ampdbfs(klevel)
	a1              chnget "SecondaryDry"
        aout            = a1*klevel        
			chnmix aout, "MasterOut2"
        ; reset channels for secondary assoc
        a0              = 0
                        chnset a0, "SecondaryDry"
			chnset a0, "SecondaryDelaySend"
			chnset a0, "SecondaryReverbSend"

	endin
