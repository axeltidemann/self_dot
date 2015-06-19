#!/usr/bin/python
# -*- coding: latin-1 -*-

#    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
#
#    This file is part of [self.]
#
#    [self.] is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 
#    as published by the Free Software Foundation.
#
#    [self.] is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with [self.].  If not, see <http://www.gnu.org/licenses/>.

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''
from altbrain import AudioSegment, AudioMemory
import utils

def transfer(NAPs, wavs, wav_audio_ids):

    audio_memory = AudioMemory()
    
    for audio_id, nappers in enumerate(NAPs):
        for i in range(len(nappers)):
            NAP = nappers[i]
            if len(NAP):
                clean_key, overlap_key = audio_memory._digitize(NAP)
                crude_hash = utils.d_hash(NAP, hash_size=8)
                fine_hash = utils.d_hash(NAP, hash_size=16)

                wav_file = wavs[audio_id][i]

                try:
                    audio_segment = AudioSegment(audio_id, crude_hash, fine_hash, wav_file, wav_audio_ids[(wav_file, audio_id)])
                except:
                    print '{} {} barfed'.format(wav_file, audio_id)
                    continue
                        
                audio_memory._insert(audio_memory.NAP_intervals, clean_key, audio_segment)
                audio_memory._insert(audio_memory.NAP_intervals, overlap_key, audio_segment)
                audio_memory._insert(audio_memory.audio_ids, audio_id, audio_segment)

    return audio_memory
