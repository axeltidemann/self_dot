; ******************************
; map channels to channels, with scaling
; use for testing how the different input parameters affect the sound directly (not going through the NN)


; ******************************
; test, for spectral synth
; get spectral profile from sound file, 
; to use as amp or freq data for selfvoice spectral synth
	instr 42
	Ssound		strget p4
	Spath		= "testsounds/"
	S1		strcat Spath, Ssound
	a1		soundin S1
	a2		= 0

	ifna		= gifnaResyn
	ifnf		= gifnfResyn
#include "audio_analyze.inc"

			outs a1, a2
	endin

; ******************************
; single parameter mapping instr 
; ADD: mapping types (log2, powoftwo, sqrt), and then postScale and postOffset

	instr 44
	Sin		strget p4
	Sout		strget p5
	iscale		= p6
	ioffset		= p7
	kval		chnget Sin
	kval		= (kval*iscale)+ioffset
			chnset kval, Sout
	endin 

; ******************************
; mapping macro
	instr 45
	imapInstr 	= 44
	ionoff		= p4
	Smap		strget p5
	inoiseband	strcmp Smap, "noiseband" 
	ipartikkel	strcmp Smap, "partikkel" 
	ispectral	strcmp Smap, "spectral" 
	print inoiseband, ipartikkel, ispectral	

	ktrig		init 1

	if inoiseband == 0 then
	Sline sprintfk {{
	i %f 0 %i "level1"    "respondLevel1"     1   0
	i %f 0 %i "pitch1pll" "respondPitch1pll"  1   0
	i %f 0 %i "centroid1" "respondCentroid1"  1   0
	}}, (imapInstr+0.1)*ionoff, -1*ionoff, (imapInstr+0.2)*ionoff, -1*ionoff, (imapInstr+0.3)*ionoff, -1*ionoff
	puts Sline, ktrig
	scoreline Sline, ktrig
	ktrig 		= 0
	endif

	if ipartikkel == 0 then
	Sline sprintfk {{
	i %f 0 %i "level1"       "partikkel1_amp"   	1   0
	i %f 0 %i "pitch1ptrack" "partikkel1_grainrate" 1   0
	i %f 0 %i "centroid1"    "partikkel1_wavfreq"  	1   0
	i %f 0 %i "autocorr1"    "partikkel1_graindur" 	1   0.1
	}}, (imapInstr+0.1)*ionoff, -1*ionoff, (imapInstr+0.2)*ionoff, -1*ionoff, (imapInstr+0.3)*ionoff, -1*ionoff, (imapInstr+0.4)*ionoff, -1*ionoff
	puts Sline, ktrig
	scoreline Sline, ktrig
	ktrig 		= 0
	endif

	if ispectral == 0 then
	kflag chnget "pvsinflag"
	tablecopy gifnaResyn,gifna
	tablecopy gifnfResyn,gifnf
	endif

/*          cSet("respondLevel1", sound[0])
            cSet("respondPitch1ptrack", sound[1])
            cSet("respondPitch1pll", sound[2])
            cSet("respondCentroid1", sound[4])
            # test partikkel generator
            cSet("partikkel1_amp", sound[0])
            cSet("partikkel1_grainrate", sound[1])
            cSet("partikkel1_wavfreq", sound[4])
            cSet("partikkel1_graindur", sound[3]+0.1)

            mic.append([cGet("level1"), 
                        cGet("pitch1ptrack"), 
                        cGet("pitch1pll"), 
                        cGet("autocorr1"), 
                        cGet("centroid1"),
*/
	endin
	

