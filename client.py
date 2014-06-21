import sys
from multiprocessing.managers import SyncManager
import uuid

import numpy as np

from utils import sleep

if __name__ == '__main__':
    class RemoteManager(SyncManager):
        pass

    RemoteManager.register('get_state')
    RemoteManager.register('get_mic')
    RemoteManager.register('get_speaker')
    RemoteManager.register('get_camera')
    RemoteManager.register('get_projector')
    manager = RemoteManager(address=(sys.argv[1], int(sys.argv[2])), authkey=sys.argv[3])
    manager.connect()

    state = manager.get_state()
    mic = manager.get_mic()
    speaker = manager.get_speaker()
    camera = manager.get_camera()
    projector = manager.get_projector()

    name = str(uuid.uuid1())

    print 'Client {} started'.format(name)

    while sleep(.1):
        audio = mic.latest(name)
        video = camera.latest(name)
        print audio.shape if len(audio) else audio, video.shape if len(video) else video
        for frame in video:
            projector.append(frame)
