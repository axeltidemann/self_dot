#!/usr/bin/python
# -*- coding: latin-1 -*-

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
