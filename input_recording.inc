;******************************
; recording  of audio input 
	instr 33
        ; moved to Python
	endin

        instr 34
        ; We write the sound file in Csound, and the text file in Python
        Swavname        strget p4
	print p1, p2, p3	
	puts Swavname, 1
        a1              chnget "rec_in1"
                	fout Swavname, 14, a1
                        
        ; for writing transient marker txt file
        ktime           timeinsts
                        chnset ktime, "memRecTimeMarker"     ; send time since start of file to Python

                        xtratim 1/kr
        krelease        release
        kactive         = 1-krelease
                        chnset kactive, "memRecActive"          ; tell Python that we're recording

        ; write pitch data to a separate file
        iPitchFn        ftgentmp 0, 0, 131072, 2, 0	; empty table
        kcps            chnget "pitch1_amdf"
        kindx		init 1
		        tablew	kcps, kindx, iPitchFn
        kindx		= kindx + 1
        ; write end index (length of data) to table index 0
		        tablew	kindx, 0, iPitchFn

        ; write table to file when instrument stops
        istrlen         strlen Swavname
        Spitchfile      strsub Swavname, 0, istrlen-4
        Spitchfile      strcat Spitchfile, "_pitch.txt"
	puts Spitchfile, 1
        ksaveonce       trigger krelease, 0.5, 0
        if ksaveonce == 1 then
                        ftsavek	Spitchfile, ksaveonce, 1, iPitchFn
        endif

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
        
