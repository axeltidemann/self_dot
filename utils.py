import os
import wave
import csv
import time
import re
import linecache
import sys
import cPickle as pickle
import zlib
import random

import numpy as np
import zmq
from scipy.io import wavfile
import scipy.fftpack
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import cv2

from brain import cochlear
findfloat=re.compile(r"[0-9.]*")


def save(filename, data):
    pickle.dump(data, file(filename, 'w'))
    print '{} saved ({})'.format(filename, filesize(filename))

def load(filename):
    data = pickle.load(file(filename, 'r'))
    print 'Part of brain loaded from file {} ({})'.format(filename, filesize(filename))
    return data

def write_cochlear(wav_file):
    array_to_csv('{}-cochlear.txt'.format(wav_file), cochlear(wav_file))

def load_cochlear(wav_file):
    return csv_to_array('{}-cochlear.txt'.format(wav_file))

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


def csv_to_array(filename, delimiter=' '):
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        return np.array([ [ float(r) for r in row ] for row in reader ])


def array_to_csv(filename, data, delimiter=' '):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        if len(data.shape) == 1:
            data.shape = (data.shape[0],1)
        for row in data:
            writer.writerow(row)


def wait_for_wav(filename):
    # Super ugly hack! Since Csound might not be finished writing to the file, we try to read it, and upon fail (i.e. it was not closed) we wait .05 seconds.
    while True:
        try:
            wavfile.read(filename)
            break
        except:
            time.sleep(.05)
    return filename
            
def filesize(filename):
    return bytes2human(os.path.getsize(filename))

def chunks(A, chunk_len):
    i = 1
    result = []
    while i*chunk_len <= A.shape[0]:
        result.append(A[chunk_len*(i-1):chunk_len*i])
        i+=1
    return result

def wav_duration(filename):
    sound = wave.open(filename, 'r')
    return sound.getnframes()/float(sound.getframerate())


def trim(A, threshold=100):
    ''' Trims off excess fat on either side of the thresholded part of the signal.'''
    right = A.shape[0]-1
    while max(A[right]) < threshold:
        right -= 1
    left = 0 
    while max(A[left]) < threshold:
        left += 1
    return A[left:right]


def trim_right(A, threshold=.2):
    ''' Trims right side of the thresholded part of the signal.'''
    maxes = np.max(A, axis=1)
    apex = np.argmax(maxes)

    for i,m in enumerate(maxes[apex:]):
        if m < threshold:
            return A[:i+apex]
    return A

        
def split_signal(data, threshold=100, length=5000, elbow_grease=100, plot=False, markers=[]):
    ''' Splits the signal after [length] silence '''
    abs_data = abs(data)
    starts = np.array(sorted([ i for i,d in enumerate(abs_data) if i > length and np.mean(d) > threshold and all(np.mean(abs_data[i-length:i], axis=1) < threshold) ] + markers)) - elbow_grease
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



def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """Pack and compress an object with pickle and zlib."""
    pobj = pickle.dumps(obj, protocol)
    zobj = zlib.compress(pobj)
    print 'Zipped pickle is {} bytes'.format(len(zobj))
    return socket.send(zobj, flags=flags)


def recv_zipped_pickle(socket, flags=0):
    """Reconstruct a python object sent with zipped_pickle"""
    zobj = socket.recv(flags)
    pobj = zlib.decompress(zobj)
    return pickle.loads(pobj)


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


def diff_to_hex_bad(diff):
    bits = "".join(map(lambda pixel: '1' if pixel else '0', diff))
    hexadecimal = int(bits, 2).__format__('016x').upper()
    return hexadecimal


# It seems as average hash is better at determining greater difference than
# perceptive hash - which is what we want, basically.
def average_hash(image):
    avg = np.mean(image)
    diff = image > avg
    return diff_to_hex(diff)


def a_hash(image, hash_size=8):
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    diff = np.ndarray.flatten(image) > np.mean(image)
    return diff_to_hex(diff)

    
def p_hash(image, hash_size=64):
    # Scale to [0,255]
    image = np.rint(scale(image)*255)

    # Remember: width x height in cv2.resize function
    image = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    dct = scipy.fftpack.dct(image)
    dctlowfreq = np.ndarray.flatten(dct[:8, 1:8])
    avg = dctlowfreq.mean()
    diff = dctlowfreq > avg
    return diff_to_hex(diff)


def diff_to_hex(difference):
    # Convert the binary array to a hexadecimal string.
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0

    return ''.join(hex_string)


def d_hash(image, hash_size = 8):
        # Scale to [0,255]
        image = np.rint(scale(image)*255)
        
        # Remember: width x height in cv2.resize function
        image = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
        # Compare adjacent pixels.
        difference = []
        for row in xrange(hash_size):
            for col in xrange(hash_size):
                pixel_left = image[row, col]
                pixel_right = image[row, col + 1]
                difference.append(pixel_left > pixel_right)

        return diff_to_hex(difference)

    
def hamming_distance(s1, s2):
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def zero_pad(signal, length):
    return np.vstack(( signal, np.zeros(( length - signal.shape[0], signal.shape[1])) )) if signal.shape[0] < length else signal


def scale(image):
    return (image - np.min(image))/(np.max(image) - np.min(image))

def getSoundInfo(wavfile):
    f = open(wavfile[:-4]+'.txt', 'r')
    segments = []
    enable = 0
    startTime = 0
    totalDur = 0
    maxAmp = 1
    for line in f:
        if 'Self. audio clip perceived at ' in line:
	        startTime = float(line[30:])
        if 'Total duration:' in line: 
            enable = 0
            totalDur = float(line[16:])
        if 'Max amp for file:' in line:
            maxAmp = float(line[18:])
        if enable:
            start,amp,pitch,centroid = line.split(' ')
            segments.append([float(start),float(amp),float(pitch),float(centroid)]) 
        if 'Sub segments (start, amp, ' in line: enable = 1
    return startTime, totalDur, maxAmp, segments


def get_segments(wavfile, threshold=.25):
    ''' Find segments in audio descriptor file. Transients closer together than the threshold will be excluded.'''
    _, totalDur, _, segments = getSoundInfo(wavfile)
    segmentTimes = []
    for item in segments:
        segmentTimes.append(item[0])    
    segmentTimes.append(totalDur)    
    return np.array(segmentTimes)
    
#    audio_info = open(wavfile[:-4]+'.txt')
#    segments = []
#    for line in audio_info:
#        if 'Total duration:' in line:
#            segments.append(float(line[16:]))
#        try:
#            segments.append(float(line))
#        except:
#            continue
#
#    remove = [ i for i,d in enumerate(np.diff(segments)) if d < threshold ]
#    for i in remove[::-1]:
#        segments.pop(i+1)
#        
#    return np.array(segments)

#def getSoundParmFromFile(responsefile):
#    audioinfo = open(responsefile[:-4]+'.txt')
#    maxamp = 1
#    dur = 1
#    for line in audioinfo:
#        if 'Total duration:' in line: 
#            dur = float(line[16:])
#        if 'Max amp for file:' in line:
#            maxamp = float(line[18:])
#    return dur, maxamp

def getLatestMemoryWavs(howmany):
    '''
    Find the N latest recorded memory wave files
    '''
    path = './memory_recordings/'
    infiles = os.listdir(path)
    wavfiles = []
    for f in infiles:
        if f[-4:] == '.wav': wavfiles.append(path+f)
    wavfiles.sort()
    latefiles = wavfiles[-howmany:]        
    return latefiles

def updateAmbientMemoryWavs(currentFiles):
    newfiles = getLatestMemoryWavs(10)
    for f in currentFiles:
        try:
            newfiles.remove(f)
        except: pass
    new = random.choice(newfiles)
    currentFiles.append(new)
    if len(currentFiles) > 4:
        currentFiles.pop(0)
    return new, currentFiles
    
def print_exception(msg=''):
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print '{} EXCEPTION IN ({}, LINE {} "{}"): {}'.format(msg, filename, lineno, line.strip(), exc_obj)
