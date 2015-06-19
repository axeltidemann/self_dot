;    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
;
;    This file is part of [self.]
;
;    [self.] is free software: you can redistribute it and/or modify
;    it under the terms of the GNU General Public License version 3 
;    as published by the Free Software Foundation.
;
;    [self.] is distributed in the hope that it will be useful,
;    but WITHOUT ANY WARRANTY; without even the implied warranty of
;    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;    GNU General Public License for more details.
;
;    You should have received a copy of the GNU General Public License
;    along with [self.].  If not, see <http://www.gnu.org/licenses/>.


        instr 90
; load ftables for instr 92, first muting the source wave to be replaced to avoid reading from a table while we relocate it

        Sfile           strget p4
        giAmbientTabNum = (giAmbientTabNum+1)%4
        kamptemp        linseg 1, 0.1, 0, 3.8, 0, 0.1, 1, 1, 1  ; swoop down to mute the source wave while we change the table underneath
        SchanName       sprintf "ambientWaveamp%i", giAmbientTabNum+1
        puts SchanName, 1
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

