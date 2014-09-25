#!/usr/bin/python
# -*- coding: latin-1 -*-

import multiprocessing as mp
import os
import numpy as np
import zmq
import IO
import time


def analyze(host):
    me = mp.current_process()
    print me.name, 'PID', me.pid

    context = zmq.Context()

    assoc = context.socket(zmq.SUB)
    assoc.connect('tcp://{}:{}'.format(host, IO.ASSOCIATIONS))
    assoc.setsockopt(zmq.SUBSCRIBE, b'')

    poller = zmq.Poller()
    poller.register(assoc, zmq.POLLIN)

    timeStamp = time.time()
    while True:
	print 'assoc is running %i', time.time()-timeStamp
	time.sleep(1)

        events = dict(poller.poll(timeout=0))
        if assoc in events:
            test = assoc.recv_json()
	    print 'test', test

if __name__ == '__main__':
    analyze('localhost')
