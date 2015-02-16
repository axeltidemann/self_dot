;******************************
; recording  of audio input 
	instr 33
        ; moved to Python
	endin

        instr 34
                
        ; We write the sound file in Csound, and the text file in Python
        Swavname        strget p4
        a1              chnget "rec_in1"

        ; release flag handling, trigger off gate also on external termination of instr instance
                        xtratim 1/kr
        kreleaseFlag    release

        ; write pitch data to a separate file
        istrlen         strlen Swavname
        Spitchfile      strsub Swavname, 0, istrlen-4
        Spitchfile      strcat Spitchfile, "_pitch.txt"

        ; create a temporary table to write segment markers to
        iMarkerTemp    ftgentmp   0, 0, 32, 2, 0     ; for writing valid segment start times (in one sentence)
                        chnset iMarkerTemp, "giMarkerTemp"

	    krec_floor	= -64                   ; safe to set statically, as the input should be gated before here anyway
	    kAttack		= 0.001			        ; envelope follower attack
	    kRelease	= 0.35                   ;chnget "in_envelopeRelease" ; 0.3 ; envelope follower release
	    aFollow1	follow2	a1, kAttack, kRelease	; envelope follower
	    kFollow1	downsamp aFollow1	
	    kFollowdb1	= dbfsamp(kFollow1)		    ; convert to dB

        krunning_time   timeinsts
        kmarker_index   init 0
        kstart          init 0
        krectime        init 0
        krms_rec_max    init -96

	    knoiseFloor_dB	chnget "inputNoisefloor"
        ikeep_thresh    = 8                ; we want to skip recorded segments with a max amplitude below this level 
                                            ; (relative to noise floor), so the effective skipping thresh is noisefloor+keep_thresh
        i_max_segment_len   = 12
        i_max_sentence_len  = 25
        k_one               = 1
        if krunning_time > i_max_sentence_len then
                        chnset k_one, "too_long_sentence"
                        turnoff2 p1, 0, 0
        endif


; compress before recording
	    kthresh         = -10
	    kratio          = 2.5
	    kattack		    = 15				; attack time 
	    krelease	    = 20				; release time 
#include "compressor.inc"

        aline          line 0, 1, 1         ; audio index for recording and re-recording, offset by running time at the point of recording start
        ktimer          init 0
        kskiptimer      init 0
        ktransient      chnget "transient"
        ;Stransient      sprintfk "*** Transient %i at time %f", ktransient, krunning_time
        ;puts Stransient, ktransient+1
        ktest           =  (kFollowdb1 > krec_floor ? 1:0)
        ;Stest           sprintfk "* Envelope over rec floor %i at time %f", ktest, krunning_time
        ;puts Stest, ktest+1
if ktransient > 0 then
reinit startgate        
endif
        irecdelay       chnget "recdelay"
startgate:
        kinit           linseg 1, irecdelay+(4/kr), 1, 0, 0, 1, 0
rireturn
        ;Sinit           sprintfk "* Initial gate open %i at time %f", kinit, krunning_time
        ;puts Sinit, kinit+1
        ktest2          = (kinit == 1 ? 1 : ktest)
        kgate           init 1
        kgate           = (kinit == 1 ? 1 : kgate)
        kgate           = (ktransient == 1 ? 1 : kgate)
        kgate           = (ktest2 < 1 ? 0 : kgate)
        ;Sgate           sprintfk "* Gate %i at time %f", kgate, krunning_time
        ;puts Sgate, kgate+1

        kgatetrig       trigger kgate, 0.5, 0
        kgatetrig_off   trigger kgate, 0.5, 1
        kgatetrig_off   += kreleaseFlag
        ;Sgatetrig       sprintfk "* Gate trigger on %i at time %f", kgatetrig, krunning_time
        ;puts Sgatetrig, kgatetrig+1
        
if kgatetrig > 0 then
        kresetindex     = krunning_time
        ksegment_start  = krectime
        kstart          = 0
endif

;;;**

if (kgate > 0) kgoto record
        kskiptimer      += 1
        kskiptime       = kskiptimer/kr     ; time in seconds (only counting when we are not recording
kgoto record_done
record:

        ktimer          += 1
        krectime        = ktimer/kr         ; time in seconds (only counting when we are recording
        aindex          = aline-kresetindex
        kindex          downsamp aindex
        if kindex > i_max_segment_len then
                        chnset k_one, "too_long_segment"
                        turnoff2 p1, 0, 0
        endif
                        tablew a1, aindex*sr, giAudioTemp
        
        kcps            chnget "pitch1_amdf"
        kcps            mediank kcps, 100, 100
                        tablew kcps, kindex*kr, giPitchTemp
                       
        kstart          += ksmps
        ;kFollowFilt     tonek kFollowdb1, 2 ; we want to integrate the max level over the whole segment, so we can lowpass filter the envelope a bit to get rid of weak spurious blips
        krms_rec        = kFollowdb1
        krms_rec_max    max, krms_rec_max, krms_rec
        ;Srecmax         sprintfk "* Recording max amp %f at time %f", krms_rec_max, krunning_time
        ;puts Srecmax, krms_rec_max
record_done:

        ;Sgateoff        sprintfk "* Gate trigger off %i at time %f", kgatetrig_off, krunning_time
        ;puts Sgateoff, kgatetrig_off+1

if kgatetrig_off > 0 then
        krms_rec_max_prev = krms_rec_max
        krms_rec_max    = -96

        kcheck  = (krms_rec_max_prev > knoiseFloor_dB+ikeep_thresh ? 1 : 0)
        ;Scheck  sprintfk "* Check if rec max for segment is over floor+thresh %i at time %f", kcheck, krunning_time
        ;puts Scheck, kcheck+1

        kstart_transfer = 0
        kmarker_index   += 1
        Smarker         sprintfk "* Marker index %i", kmarker_index
  if kcheck > 0 then
writeFile:
        atemp           tablera giAudioTemp, kstart_transfer, 0
                        fout Swavname, 14, atemp
        ktemp           table kstart_transfer/ksmps, giPitchTemp
                        dumpk kcps, Spitchfile, 7, 0

        kstart_transfer += ksmps
    if kstart_transfer < kstart then
        kgoto writeFile
    endif
  else
    ;Sskipwrite          sprintfk "Writing skipsegment indicator at time %f", krunning_time
    ;puts Sskipwrite, krunning_time
                        tablew -1, kmarker_index-1, iMarkerTemp ; handle this in python, if zero, delete segment
  endif
endif


; write all, unsegmented
        Swavname1       strcat Swavname, "ALL.wav"
                	    fout Swavname1, 14, a1

                        
        ; for writing transient marker txt file
                        chnset krectime, "memRecTimeMarker"     ; time of recorded sound
                        chnset kskiptime, "memRecSkiptime"     ; skip time (amount of silence stripped from segment)

        ktransient      chnget "transient"
if ktransient > 0 then  ; reset skiptimer on each transient
        kskiptimer      = 0
endif




                        xtratim 1/kr
        kreleaseFlag    release

        ; save max amp for this sound file so we can normalize each segment on the fly later
        kmetro          metro 5
        kmaxa           max_k a1, kmetro, 1
        kmaxamp         init 0
        kmaxamp         max, kmaxamp, kmaxa
                        chnset kmaxamp, "memRecMaxAmp"            

        kactive         = 1-kreleaseFlag
                        chnset kactive, "memRecActive"          ; tell Python that we're recording



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
        
