;******************************
; recording  of audio input 
	instr 33
        ; moved to Python
	endin

        instr 34
        ; We write the sound file in Csound, and the text file in Python
        Swavname        strget p4
        a1              chnget "rec_in1"

	    krec_floor	= -65                   ; safe to set statically, as the input should be gated before here anyway
	    kAttack		= 0.001			        ; envelope follower attack
	    kRelease	= 0.25                   ;chnget "in_envelopeRelease" ; 0.3 ; envelope follower release
	    aFollow1	follow2	a1, kAttack, kRelease	; envelope follower
	    kFollow1	downsamp aFollow1	
	    kFollowdb1	= dbfsamp(kFollow1)		    ; convert to dB


; compress before recording
	    kthresh         = -15
	    kratio          = 2.5
	    kattack		    = 15				; attack time 
	    krelease	    = 20				; release time 
#include "compressor.inc"

        ktimer          init 0
        kskiptimer      init 0

ktest =  (kFollowdb1 > krec_floor ? 1:0)
ktest2 = (ktimer == 0 ? 1 : ktest)
printk2 ktest2

if (ktest2 > 0) kgoto record
        kskiptimer      += 1
        kskiptime       = kskiptimer/kr     ; time in seconds (only counting when we are not recording
kgoto record_done
record:
        ktimer          += 1
        krectime        = ktimer/kr         ; time in seconds (only counting when we are recording
                	    fout Swavname, 14, a1
record_done:

        ;printk2 krectime, 5
                        
        ; for writing transient marker txt file
                        chnset krectime, "memRecTimeMarker"     ; time of recorded sound
                        chnset kskiptime, "memRecSkiptime"     ; skip time (amount of silence stripped from segment)

        ktransient      chnget "transient"
if ktransient > 0 then  ; reset skiptimer on each transient
        kskiptimer      = 0
endif

                        xtratim 1/kr
        krelease        release

        ; save max amp for this sound file so we can normalize each segment on the fly later
        kmetro          metro 5
        kmaxa           max_k a1, kmetro, 1
        kmaxamp         init 0
        kmaxamp         max, kmaxamp, kmaxa
                        chnset kmaxamp, "memRecMaxAmp"            

        kactive         = 1-krelease
                        chnset kactive, "memRecActive"          ; tell Python that we're recording

        ; write pitch data to a separate file
        istrlen         strlen Swavname
        Spitchfile      strsub Swavname, 0, istrlen-4
        Spitchfile      strcat Spitchfile, "_pitch.txt"
        ;puts Spitchfile, 1
        kcps            chnget "pitch1_amdf"
        kcps            mediank kcps, 100, 100
                        dumpk kcps, Spitchfile, 7, 0


/*
	; trigger instr events to record subsegments
	kcount		init 0
	kmetro		metro 2
	kcount		= kcount + kmetro
	Sinstr		sprintfk "i 35 0 0.5 %i", kcount
	if kmetro > 0 then 
	reinit makestring
	endif
	igoto end
	makestring:
	Sevent		strcat Sinstr, "\""
	Sevent		strcat Sevent, Swavname
	Sevent		strcat Sevent, "\""
			scoreline_i Sevent
	end:
*/
        endin

	instr 35
        ; write subsegments of the sound file, to speed up classification/analysis
	icount		= p4
        Swavname        strget p5
	Ssegment	sprintf "segment_%i_", icount
	Spath		= "./memory_recordings/"	
	ilenpath	strlen Spath
	Sfile		strsub 	Swavname, ilenpath, -1
	Spath		strcat Spath, Ssegment
	Spath		strcat Spath, Sfile
        a1              chnget "rec_in1"
                	fout Spath, 14, a1

	endin
        
