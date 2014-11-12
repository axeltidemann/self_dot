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


;#include "cs_python_globals.inc"
#include "ftables.inc"
#include "udos.inc"

;******************************
; DEBUG
	instr 1
k1 init 0
k1 = (k1+1)%100
printk2 k1
        endin

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
; instr 50-59
; primary synth, resynthesis/imitation
;#include "synth_primary.inc"

; ******************************
; instr 60-69
; self's voice instruments
#include "self_voices.inc"

; ******************************
; instr 70-79
; Playback of secondary associations / memory images of heard sounds
; #include "synth_secondary.inc"
#include "secondary_efx.inc"
; ******************************
; instr 91-99
; Master output, self analysis
#include "master_outputs.inc"

;******************************
; DEBUG
	instr 100
k1 init 0
k1 = (k1+1)%100
printk2 k1+9900, 30
        endin

</CsInstruments>

<CsScore>
; run for N sec
#define SCORELEN # 604800 # ; 604800 secs is one week
;#define SCORELEN # 15 #

f0 86400

;i1   0 $SCORELEN              ; debug
i2 	 0 .1			        ; init chn values
i4 	 0 $SCORELEN			; audio input
i11  0 $SCORELEN			; panalyze, merge left and right input
i21  0 .1 1				; initialize input level
i22  0 .1 "inputNoisefloor" -20	; initialize noise floor 
i31  0 $SCORELEN			; analysis
i77  0 $SCORELEN			; delay for secondary associations playback
i78  0 $SCORELEN			; reverb for secondary associations playback
i79  0 $SCORELEN			; mixer for secondary associations playback
i93  0 $SCORELEN			; ambient sound reverb
i99  0 $SCORELEN			; master out
;i100 0 $SCORELEN              ; debug

</CsScore>

</CsoundSynthesizer>
