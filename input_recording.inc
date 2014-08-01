;******************************
; recording  of audio input 
; instr 33 initiate events for instr 34
	instr 33
        krecord         chnget "memoryRecording"
        kstatusTrig     chnget "audioStatusTrig"
        
        if (krecord > 0) && (kstatusTrig > 0) then
                        event "i", 34, 0, -1        
        endif
        if kstatusTrig < 0 then
                        event "i", -34, 0, 1        
        endif
        
	endin

        instr 34
        ; We write the sound file in Csound, and the text file in Python,
        ; since need the same base file name in both places we have some awkward manoeuvres here
                        pyruni "import time"
                        pyruni "tim = time.time()"
                        pyruni "t = time.gmtime(tim)"
        iyear           pyevali "float(t[0])"
        imonth          pyevali "float(t[1])"
        iday            pyevali "float(t[2])"
        ihour           pyevali "float(t[3])"
        iminute         pyevali "float(t[4])"
        isecond         pyevali "float(t[5])"
        itime           pyevali "tim"
        ifrac           = int(frac(itime)*10)*0.1
        Stimedate       sprintf "../memoryRecording/%i_%i_%i_%i_%i_%i_%.1f", iyear, imonth, iday, ihour, iminute, isecond, ifrac
        Swavname        strcat Stimedate, ".wav"
        Stxtname        strcat Stimedate, ".txt"
        Stxtnamepy      strcat "txtfilename = '", Stxtname
        Stxtnamepy      strcat Stxtnamepy, "'"
                        pyruni Stxtnamepy
                        pyruni "f = open(txtfilename, 'w')"
                        pyruni "f.write('Self. audio clip perceived at %s'%txtfilename[19:-4])"
                        pyruni "segments = '\\nSub segment start times: \\n0.000 \\n'"
        ; write audio file
        a1              chnget "rec_in1"
                	fout Swavname, 14, a1

        ; write metadata txt file
        ktransient      chnget "transient"
        ktime           timeinsts
        if ktransient > 0 then
                        SegAdd  sprintfk "segments += str(%.3f) +' \\n'", ktime
                        pyrun SegAdd
        endif
                        xtratim 1/kr
        krelease        release
        if krelease > 0 then
                        pyrun "f.write(segments)"
                        Stotal  sprintfk "total = '\\nTotal duration: %f'", ktime
                        pyrun Stotal
                        pyrun "print 'test'"
                        pyrun "print total"
                        pyrun "f.write(total)"
                        pyrun "f.close()"
        endif
        endin
        
                instr 35
        ; TEST
                        pyruni "import time"
                        pyruni "print 'Python is working inside Csound hahahahahahahaha'"
                endin
