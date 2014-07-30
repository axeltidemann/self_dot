; ******************************
; read audio segments from memoryRecording
        instr 70
        ; use python and ._CSOUND_ to get the file names and initiate events of instr 71 and 72

        pyruni {{
files = os.listdir(memoryPath)
# simple test
#loadAudioAndMarkers(random.choice(getBasenames()))
testname = getBasenames(files)[1]
loadAudioAndMarkers(testname)
#soundfileGC(-1) # GC immediately

}}
        endin

        instr 71
; load sound file and corresonding transient markers
	itabNum		= p4
        Swavname        strget p5
        iSound          ftgen itabNum, 0, 0, 1, Swavname, 0, 0, 0

        Smarkers        strget p6
        iTotdur         = p7
        iMarkers        ftgen itabNum+100, 0, 1024, 7, -1, 1024, -1 ; we can use -1 as eof, the first address not explicitly set will now contain -1
                        tableiw  iTotdur, 0, iMarkers
        index           = 1
        parseMarkers:
        ipos            strindex Smarkers, " "
        Snum            strsub Smarkers, 0, ipos
        Smarkers        strsub  Smarkers, ipos+1, -1        
        inum            strtod Snum
                        tableiw  inum, index, iMarkers
        index           += 1
        ilen            strlen Smarkers
        if ilen > 1 then
        igoto parseMarkers
        endif

        endin

        instr 72
; free sound file and marker table
	itabNum		= p4
        Sbasename        strget p5
        puts Sbasename, 1

        ; we might want to check if the table is being used by any secondary synth before deleting (fixme on this)
        itabInUse       = 0
        if itabInUse == 0 then
                        ftfree itabNum, 0       ; sound file
                        ftfree itabNum+100, 0   ; markers
        Spystring       sprintf "del filesloaded['%s']", Sbasename
        pyruni "print filesloaded"
        pyruni Spystring
        endif
        endin

; ******************************
; straight playback of audio segments
        instr 74
        itab            = p4
                        tableiw 1, itab, giTablesInUseSecondary ; indicate that the table is in use
        idur            = nsamp(itab)/sr
        p3              = idur
        a1              loscil3 1, 1, itab, 1

#include "synth_secondary_out_and_release.inc"
        endin

; ******************************
; partikkel instr 
	instr 75
        itab            = p4
                        tableiw 1, itab, giTablesInUseSecondary ; indicate that the table is in use

;; FIXME 
;;      (USE tab as partikkel source)
;;      (Use time stretch to create slow moving playback of secondary associations)
;;      (this may require setting a longer release time, letting playback gently fade away when next secndary association is triggered)

#include "partikkel2_chn.inc"
#include "partikkel2_self.inc"

#include "synth_secondary_out_and_release.inc"
	endin

; ******************************
; spectral synth playback of audio segments
        instr 76
        itab            = p4
                        tableiw 1, itab, giTablesInUseSecondary ; indicate that the table is in use

;; FIXME (PLAYBACK METHOD)
        a1              = 0
#include "synth_secondary_out_and_release.inc"
        endin

; ******************************
; delay for secondary assoc 
	instr 77
	a1              chnget "SecondaryDelaySend"
        kTime		chnget "SecondaryDelayTime"
        kFeed		chnget "SecondaryDelayFeed"
	kFiltFq		chnget "SecondaryDelayFiltFq"
        imaxdel         = 2
	adelTim		upsamp 	kTime				; smoothing
	adelTim		tone	adelTim, 2			; smoothing
	adummy		delayr imaxdel				; establish delay line
	aout		deltapi adelTim				; tap delay 
	adelw		= a1 + (aout*kFeed)			; mix input and feedback
	adelw		butterlp adelw, kFiltFq			; filter delay signal
	adelw           butterhp adelw, 40                      ; low cut on feedback
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
        kLoFreq         chnget "SecondaryMasterLoCut"
        aout            butterhp aout, kLoFreq        
			chnmix aout, "MasterOut2"
        ; reset channels for secondary assoc
        a0              = 0
                        chnset a0, "SecondaryDry"
			chnset a0, "SecondaryDelaySend"
			chnset a0, "SecondaryReverbSend"

	endin
