<CsoundSynthesizer>
<CsOptions>
;-odac1 -iadc0 
;-odac18 -iadc20
; -iadc -d
</CsOptions>

<CsInstruments>

	sr = 44100  
	ksmps = 256
	nchnls = 2	
	0dbfs = 1


;#include "cs_python_globals.inc"
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
; instr 50-59
; primary synth, resynthesis/imitation
#include "synth_primary.inc"

; ******************************
; instr 60-69
; self's voice instruments
#include "self_voices.inc"

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
;#define SCORELEN # 15 #

;#include "testscore.inc"

i1 	0 .1			        ; init chn values
i4 	0 $SCORELEN			; audio input
i11 	0 $SCORELEN			; panalyze, merge left and right input
i21 	0 .1 1				; initialize input level
i22 	0 .1 "inputNoisefloor" -20	; initialize noise floor 
;i22 	0 .1 "memoryRecording" 1	; enable/disable recording of audio memory
i31 	0 $SCORELEN			; analysis
;i51 	0 -1				; subtractive harmonic resynthesis
;i52 	0 -1				; partikkel resynthesis
;i53 	3 -1				; fft resynthesis
;i98 	0 $SCORELEN			; analysis of own output
i77     0 $SCORELEN			; delay for secondary associations playback
i78     0 $SCORELEN			; reverb for secondary associations playback
i79     0 $SCORELEN			; mixer for secondary associations playback
i99 	0 $SCORELEN			; master out

; test
;i35 2 1                                ; related to osx python problems
;i2      4 1                             ; exit Csound
;i 70    1 1                             ; test read a segment from memoryRecording       
;i 74    2 1 800                         ; test play a loaded segment from memoryRecording       
;i 55    2 0 "memory_recordings/2014_08_07_15_15_54.wav"   ; test play granular file playback

/*
; test self voice
i 60    0  1   "memory_recordings/2014_09_25_10_43_46.wav"      0       0     -2    1      -96    -96    1
;          p3  soundfile                                        start   end   amp   voice   dly   rvb   speed
i 60    0  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 61    5  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 62   10  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 63   15  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 64   20  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 65   25  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 66   30  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
i 67   35  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -12    -23    1
*/
/*
; test self suppression
i 12 1 4') # measure roundtrip latency
i 13 5 1.9') # get audio input noise print
i 14 7 -1 1 1') # enable noiseprint and self-output suppression
i 15 7.2 2') # get noise floor level 
i 16 8.3 0.1') # set noise gate shape
i 17 8.5 -1') # turn on new noise gate
i 60 10  1   "memory_recordings/2014_08_07_15_15_54.wav"      0       0     -2    2      -16    -16    1
*/
</CsScore>

</CsoundSynthesizer>
