import os
import wave

import numpy as np
import zmq

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

def wav_duration(filename):
    sound = wave.open(filename, 'r')
    return sound.getnframes()/float(sound.getframerate())

def trim(A, threshold=100):
    ''' Trims off excess fat on either side of the thresholded part of the signal '''
    right = A.shape[0]-1
    while A[right] < threshold:
        right -= 1
    left = 0 
    while A[left] < threshold:
        left += 1
    return A[left:right]

def split_signal(data, threshold=100, length=5000, elbow_grease=100, plot=False, markers=[]):
    ''' Splits the signal after [length] silence '''
    abs_data = abs(data)
    starts = np.array(sorted([ i for i,d in enumerate(abs_data) if i > length and d > threshold and all(abs_data[i-length:i] < threshold) ] + markers)) - elbow_grease
    chunks = [ data[q:s] for q,s in zip(starts[:-1], starts[1:]) ]
    chunks.append(data[starts[-1]:])

    if plot:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.figure()
        plt.plot(data)
        for s in starts:
            plt.axvline(s, color='r')
        plt.axhline(threshold, color='g')
        plt.axhline(-threshold, color='g')

    return chunks

def split_wav(filename, threshold=100, length=5000, elbow_grease=100, plot=False):
    from scipy.io import wavfile
    rate, data = wavfile.read(filename)
    return split_signal(data, threshold=threshold, length=length, elbow_grease=elbow_grease, plot=plot)

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
