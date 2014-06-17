from multiprocessing.managers import SyncManager

import numpy as np

from utils import MyManager, MyDeque, sleep

if __name__ == '__main__':

    class ServerManager(SyncManager):
        pass
    
    MyManager.register('deque', MyDeque)

    manager = MyManager()
    manager.start()

    dq = manager.deque()
    
    ServerManager.register('get_deque', callable=lambda: dq)
    server_manager = ServerManager(address=('', 7777), authkey='tullball')
    server_manager.start()

    print 'Server started'

    i = 0
    while sleep(1):
        dq.append(i)
        i += 1
        print dq.array()
        
