        instr 90
; load ftables for instr 92, first muting the source wave to be replaced to avoid reading from a table while we relocate it

        Sfile           strget p4
        giAmbientTabNum = (giAmbientTabNum+1)%4
        kamptemp        linseg 1, 0.1, 0, 3.8, 0, 0.1, 1, 1, 1  ; swoop down to mute the source wave while we change the table underneath
        SchanName       sprintf "ambientWaveamp%i", giAmbientTabNum
                        chnset kamptemp, SchanName
        Scoreline       sprintf {{i 91 2 1 %i "%s"}}, giAmbientTabNum, Sfile
                        puts Scoreline, 1
                        scoreline_i Scoreline
        endin

        instr 91
        print p4
        islot           = p4
        Sfile           strget p5
        puts Sfile, 1

if islot == 0 then
        i91             ftgen       91, 0, 0, 1, Sfile, 0, 0, 0

elseif islot == 1 then
        i92             ftgen       92, 0, 0, 1, Sfile, 0, 0, 0

elseif islot == 2 then
        i93             ftgen       93, 0, 0, 1, Sfile, 0, 0, 0

elseif islot == 3 then
        i94             ftgen       94, 0, 0, 1, Sfile, 0, 0, 0
endif

        endin
    
; ***

