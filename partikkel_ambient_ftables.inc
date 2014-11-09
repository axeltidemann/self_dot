
; load ftables for instr 91
    Sfile           strget p4
    puts Sfile, 1
    giAmbientTabNum = (giAmbientTabNum+1)%8
    print giAmbientTabNum
if giAmbientTabNum == 0 then
    i91         ftgen       91, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      95, 1
                tablew 91, 0, giAmbientTabs
elseif giAmbientTabNum == 1 then
    i92         ftgen       92, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      96, 1
                tablew 92, 1, giAmbientTabs
elseif giAmbientTabNum == 2 then
    i93         ftgen       93, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      97, 1
                tablew 93, 2, giAmbientTabs
elseif giAmbientTabNum == 3 then
    i94         ftgen       94, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      98, 1
                tablew 94, 3, giAmbientTabs
elseif giAmbientTabNum == 4 then
    i95         ftgen       95, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      91, 1
                tablew 95, 0, giAmbientTabs
elseif giAmbientTabNum == 5 then
    i96         ftgen       96, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      92, 1
                tablew 96, 1, giAmbientTabs
elseif giAmbientTabNum == 6 then
    i97         ftgen       97, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      93, 1
                tablew 97, 2, giAmbientTabs
elseif giAmbientTabNum == 7 then
    i98         ftgen       98, 0, 0, 1, Sfile, 0, 0, 0
                ;ftfree      94, 1
                tablew 98, 3, giAmbientTabs
endif


