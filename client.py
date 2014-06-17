import sys
from multiprocessing.managers import SyncManager

import numpy as np

from utils import sleep

if __name__ == '__main__':
    class RemoteManager(SyncManager):
        pass

    RemoteManager.register('get_state')
    RemoteManager.register('get_projector')
    manager = RemoteManager(address=(sys.argv[1], int(sys.argv[2])), authkey=sys.argv[3])
    manager.connect()

    remote_state = manager.get_state()
    remote_projector = manager.get_projector()

    print 'Client started'

    i = 0
    while sleep(.1):
        i+=1
        Z = np.zeros(160*90)
        Z[i] = 1
        remote_projector.append(Z)
        if i == 160*90:
            i = 0
        
