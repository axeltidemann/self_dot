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



;i3 	1 8  "../../../efx_experiments/fox.wav"		; audio file input
;i3 	10 8 "../../../efx_experiments/Bach.wav"		; audio file input
;i3 	20 8 "testsounds/dakata4.wav"		; audio file input

; long test
/*
i3 	1 900 "../../../testsounds_talk/N2_010510_NRK_HON_NO.wav"		; audio file input
i3 	905 900 "../../../testsounds_talk/N2_030505_NRK_D12_NO.wav"		; audio file input
i3 	1810 900 "../../../testsounds_talk/N2_030508_NRK_POK_NO.wav"		; audio file input
i3 	2715 900 "../../../testsounds_talk/N2_030510_NRK_UKS_NO.wav"		; audio file input
i3 	3620 900 "../../../testsounds_talk/N2_030510_NRK_VPL_NO.wav"		; audio file input
i3 	4525 900 "../../../testsounds_talk/N2_030911_NRK_SDA_NO.wav"		; audio file input
#define SCORELEN # 5430 #
*/

; short test
i3 	1 97 "testsounds_talk/N2_010510_NRK_HON_NO.wav"		; audio file input
i3 	100 97 "testsounds_talk/N2_030505_NRK_D12_NO.wav"		; audio file input
i3 	200 97 "testsounds_talk/N2_030508_NRK_POK_NO.wav"		; audio file input
i3 	300 97 "testsounds_talk/N2_030510_NRK_UKS_NO.wav"		; audio file input
i3 	400 97 "testsounds_talk/N2_030510_NRK_VPL_NO.wav"		; audio file input
i3 	500 97 "testsounds_talk/N2_030911_NRK_SDA_NO.wav"		; audio file input
#define SCORELEN # 600 #
