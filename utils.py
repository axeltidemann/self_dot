from multiprocessing.managers import SyncManager
from collections import deque
import time
import os

import numpy as np

class MyManager(SyncManager):
    pass

class MyDeque(deque):
    def __init__(self, *args, **kwargs):
        deque.__init__(self, *args, **kwargs)
        self.i = 0
    
    def array(self):
        return np.array(list(self))

    def latest(self):
        data = self.array()
        old_i = self.i
        self.i = len(data)
        return data[old_i:]

    def clear(self):
        self.i = 0
        deque.clear(self)
        

def net_rmse(brain, signals):
    import Oger
    rmse = []

    for net, scaler in brain:
        scaled_signals = scaler.transform(signals)
        rmse.append(Oger.utils.rmse(net(scaled_signals[:-1]), scaled_signals[1:]))

    return rmse

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
