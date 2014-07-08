import sys
import uuid
import time

import numpy as np
import zmq

from utils import sleep, recv_array, send_array
from IO import CAMERA, PROJECTOR, MIC, SPEAKER

if __name__ == '__main__':
    context = zmq.Context()

    subscriber = context.socket(zmq.SUB)
    subscriber.connect('tcp://localhost:{}'.format(CAMERA))
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')

    publisher = context.socket(zmq.PUB)
    publisher.connect('tcp://localhost:{}'.format(PROJECTOR))

    name = str(uuid.uuid1())

    print 'Echo client {} started'.format(name)

    if len(sys.argv) == 1:
        while True:
            send_array(publisher, recv_array(subscriber))
    else:
        while True:
            send_array(publisher, abs(1-recv_array(subscriber)))
