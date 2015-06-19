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


; tail end of all secondary synth instruments
; providing outputs and release (unmarking of table used for playback)

        kDryLevel       chnget "SecondaryDryLevel"
        kDlySend        chnget "SecondaryDelaySendLevel"
        kRvbSend        chnget "SecondaryReverbSendLevel"
        aDry            = a1 * ampdbfs(kDryLevel)
        aDelaySend      = a1 * ampdbfs(kDlySend)
        aReverbSend     = a1 * ampdbfs(kRvbSend)
			chnmix aDry, "SecondaryDry"
			chnmix aDelaySend, "SecondaryDelaySend"
			chnmix aReverbSend, "SecondaryReverbSend"

; release, unmarking the table used for playback 
                        xtratim 2/kr
        krelease        release
        if krelease > 0 then
        kzero           = 0
                        tablew kzero, itab, giTablesInUseSecondary ; indicate that the table is not in use any longer
        endif
        kinuse          table itab, giTablesInUseSecondary

