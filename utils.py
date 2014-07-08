from collections import deque
import time
import os

import numpy as np
import zmq

class MyDeque(deque):
    def __init__(self, *args, **kwargs):
        deque.__init__(self, *args, **kwargs)
        self.i = np.zeros(0, dtype=np.int)
        self.markers = dict()
    
    def array(self):
        return np.array(list(self))

    def latest(self, key):
        if key not in self.markers:
            self.set_mark(key)
            return np.zeros(0)
        elif self.i[self.markers[key]] < -1:
            data = self.array()
            data = data[self.i[self.markers[key]]:]
            self.i[self.markers[key]] = 0        
            return data
        return np.zeros(0)

    def append(self, x):
        self.i -= 1
        deque.append(self, x)

    def clear(self):
        self.i *= 0

    def set_mark(self, key):
        if key not in self.markers:
            self.markers[key] = len(self.i)
            self.i = np.concatenate((self.i, np.zeros(1, dtype=np.int)))
        else:
            self.i[self.markers[key]] = 0

    def get_marked_segment(self, start, stop):
        data = self.array()
        return data[self.i[self.markers[start]]:self.i[self.markers[stop]]]

    
def signal_rmse(net, scaler, signals):
    import Oger
    scaled_signals = scaler.transform(signals)
    return Oger.utils.rmse(net(scaled_signals[:-1]), scaled_signals[1:])


def get_networks(state):
    return [ key for key in state.keys() if key.startswith('NEURAL') ]


def reset_rmses(state):
    for net in get_networks(state):
        state[net] = 'RESET'

        
def find_winner(state):
    min_rmse = np.inf
    winner = ''
    for net in get_networks(state):
        if state[net] < min_rmse:
            min_rmse = np.mean(state[net])
            winner = net
    state['respond'] = winner


# http://goo.gl/zeJZl
def bytes2human(n, format="%(value)i%(symbol)s"):
    """
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    """
    symbols = ('b', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)

def filesize(filename):
    return bytes2human(os.path.getsize(filename))

def sleep(seconds):
    time.sleep(seconds)
    return True

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = buffer(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])
