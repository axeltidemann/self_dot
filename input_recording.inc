;******************************
; recording  of audio input 
	instr 33
        ; moved to Python
	endin

        instr 34
        ; We write the sound file in Csound, and the text file in Python
        Swavname        strget p4
	    ;print p1, p2, p3	
	    ;puts Swavname, 1
        a1              chnget "rec_in1"
        ;idel            chnget "statusRel"
        ;a1del           delay a1, idel
        
; compress before recording
	kthresh         = -20
	kratio          = 4
	kattack		= 15				; attack time 
	krelease	= 20				; release time 
#include "compressor.inc"

;timout skiprec, idel
                	    fout Swavname, 14, a1
;skiprec:
                        
        ; for writing transient marker txt file
        ktime           timeinsts
                        chnset ktime, "memRecTimeMarker"     ; send time since start of file to Python

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
        
