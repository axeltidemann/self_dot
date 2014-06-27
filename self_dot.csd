<CsoundSynthesizer>
<CsOptions>
;-odac1 -iadc0 
;-odac18 -iadc20
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 128
	nchnls = 2	
	0dbfs = 1

        pyinit          ; we will call Python from Csound

#include "ftables.inc"
#include "udos.inc"

;******************************
; audio input and housekeeping
; instruments 1-29 
#include "input_housekeep.inc"

; ******************************
; instr 31 - 35
; input analysis and recording
#include "input_analyze.inc"
#include "input_recording.inc"

; ******************************
; instr 41-49
; mapping and tests
#include "channel_mapping.inc"

; ******************************
; instr 51-59
; primary synth, resynthesis/imitation
#include "synth_primary.inc"

; ******************************
; instr 70-79
; Playback of secondary associations / memory images of heard sounds
#include "synth_secondary.inc"

; ******************************
; instr 91-99
; Master output, self analysis
#include "master_outputs.inc"

</CsInstruments>

<CsScore>
; run for N sec
#define SCORELEN # 86400 #

;#include "testscore.inc"
/*
i1 	0 .1			        ; init chn values
i4 	0 $SCORELEN			; audio input
i11 	0 $SCORELEN			; merge left and right input
i21 	0 .1 1				; initialize input level
i22 	0 .1 "inputNoisefloor" -20	; initialize noise floor
i22 	0 .1 "memoryRecording" 1	; enable/disable recording of audio memory
i31 	0 $SCORELEN			; analysis
;i51 	0 -1				; subtractive harmonic resynthesis
i52 	0 -1				; partikkel resynthesis
;i53 	3 -1				; fft resynthesis
;i98 	0 $SCORELEN			; analysis of own output
i99 	0 $SCORELEN			; master out

; test
;i2      4 1                             ; exit Csound
*/
i 70 1 1
</CsScore>

</CsoundSynthesizer>
