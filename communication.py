#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Very simple communication module for [self.]

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL

Talk to [self.] over ØMQ sockets.
'''

import sys

from IO import send

if __name__ == '__main__':
    if len(sys.argv) > 1:
        send(' '.join(sys.argv[1:]))
