import numpy as np

from utils import MyManager, sleep

if __name__ == '__main__':
    MyManager.register('get_deque')
    manager = MyManager(address=('localhost', 7777), authkey='tullball')
    manager.connect()

    local_deque = manager.get_deque()

    print 'Client started'

    while sleep(1):
        print local_deque.array()
        if np.random.rand() < .3:
            local_deque.clear()
            print 'ARMAGEDDON'
