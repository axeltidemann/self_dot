#!/usr/bin/python
# -*- coding: latin-1 -*-

'''Very simple communication module for [self.]

@author: Axel Tidemann
@contact: axel.tidemann@gmail.com
@license: GPL

Talk to [self.] over sockets.
'''

from __future__ import print_function

import os
import sys
import socket

def receive(callback, host='localhost', port=7777):
    print('RECEIVE PID {}'.format(os.getpid()))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)

    print('Communication channel listening on {}:{}'.format(host, port))
    while True:
        try:
            connection, address = sock.accept()
            data = connection.recv(1024)
            callback(data)
            connection.close()
        except:
            print('Communication channel going down.')
            sock.close()
            break
    
def send(message, host='localhost', port=7777):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.sendall(message)
    sock.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for message in sys.argv[1:]:
            send(message)
    else:
        receive(print)

