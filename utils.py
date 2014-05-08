from multiprocessing.managers import SyncManager
from collections import deque

import numpy as np

class MyManager(SyncManager):
    pass

class MyDeque(deque):
    def array(self):
        return np.array(list(self))

def net_rmse(brain, signals):
    import Oger
    rmse = []

    for net, scaler in brain:
        scaled_signals = scaler.transform(signals)
        rmse.append(Oger.utils.rmse(net(scaled_signals[:-1]), scaled_signals[1:]))

    return rmse
