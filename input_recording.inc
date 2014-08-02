;******************************
; recording  of audio input 
	instr 33
        ; moved to Python
	endin

        instr 34
        ; We write the sound file in Csound, and the text file in Python,
        Swavname        strget p4
        a1              chnget "rec_in1"
                	fout Swavname, 14, a1
                        
        ; for writing transient marker txt file
        ktime           timeinsts
                        chnset ktime, "memRecTimeMarker"     ; send time since start of file to Python

                        xtratim 1/kr
        krelease        release
        kactive         = 1-krelease
                        chnset kactive, "memRecActive"          ; tell Python that we're recording
               
        endin
        
