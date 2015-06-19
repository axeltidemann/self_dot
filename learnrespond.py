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

import time
import sys

from IO import send

secs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
mode = (sys.argv[2]) if len(sys.argv) >= 3 else 'both'
autowait = 5

if mode == 'filelearn':
    send('playfile {}'.format(sys.argv[3]))
    print 'learning'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
    print 'done learning'
elif mode == 'filerespond':
    send('playfile {}'.format(sys.argv[3]))
    print 'responding'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('respond')
elif mode == 'both':
    print 'learning'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
    print 'done learning'
    time.sleep(autowait)
    print 'responding'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('respond')
elif mode == 'learn':
    print 'learning'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
    print 'done learning'
elif mode == 'respond':
    print 'responding'
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('respond')
elif mode == 'association':
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('setmarker')
    time.sleep(autowait)
    send('startrec')
    time.sleep(secs)
    send('stoprec')
    send('learn')
elif mode == 'longshort':
    send('startrec')
    time.sleep(7)
    send('stoprec')
    send('setmarker')
    time.sleep(autowait)
    send('startrec')
    time.sleep(2)
    send('stoprec')
    send('learn')
elif mode == 'shortlong':
    send('startrec')
    time.sleep(2)
    send('stoprec')
    send('setmarker')
    time.sleep(autowait)
    send('startrec')
    time.sleep(7)
    send('stoprec')
    send('learn')

else:
    print 'unknown mode', mode
