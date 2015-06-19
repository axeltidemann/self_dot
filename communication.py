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

'''Very simple communication module for [self.]

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL

Talk to [self.] over ØMQ sockets.
'''

import sys
import zmq

# Setup so it can be accessed from processes which don't have a zmq context, i.e. for one-shot messaging.
# Do not use this in contexts where timing is important, i.e. create a proper socket similar to this one.
def send(message, context=None, host='localhost', port=5566):
    print 'This send() should only be used in simple circumstances, i.e. not in something that runs in performance-critical code!'
    context = context or zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, port))
    sender.send_json(message)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        send(' '.join(sys.argv[1:]))
