import sys
from multiprocessing.managers import SyncManager

import numpy as np

from utils import sleep

if __name__ == '__main__':
    class RemoteManager(SyncManager):
        pass

    RemoteManager.register('get_state')
    manager = RemoteManager(address=(sys.argv[1], int(sys.argv[2])), authkey=sys.argv[3])
    manager.connect()

    remote_state = manager.get_state()

    print 'Client started'

    while sleep(1):
        print remote_state
