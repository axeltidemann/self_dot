#!/usr/bin/python
# -*- coding: latin-1 -*-

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
import multiprocessing as mp
import threading
import glob
from subprocess import call
import fcntl
import json
import sched
import datetime
import mmap
import subprocess
from collections import namedtuple

import numpy as np
import zmq
from scipy.io import wavfile
import scipy.fftpack
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import cv2
from scipy.stats import itemfreq

import myCsoundAudioOptions
import zmq_ports

MotorCommand = namedtuple('MotorCommand', ['robohead', 'mode', 'x_diff', 'y_diff'])
_Rectangle = namedtuple('_Rectangle', ['x', 'y', 'w', 'h'])

class Rectangle(_Rectangle):
    @property
    def center(self):
        return (self.x + self.w/2, self.y + self.h/2)

    @property
    def topleft(self):
        return (self.x, self.y)

    @property
    def bottomright(self):
        return (self.x+self.w,self.y+self.h)

NUMBER_OF_BRAINS = 5
PROCESS_TIME_OUT = 5*60 
SYSTEM_TIME_OUT = 30*60 

findfloat=re.compile(r"[0-9.]*")
find_filename = re.compile('[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+\.wav')

DREAM_HOUR = 23
EVOLVE_HOUR = 4
SAVE_HOUR = 5
REBOOT_HOUR = 6

def save(filename, data):
    pickle.dump(data, file(filename, 'w'))
    print '{} saved ({})'.format(filename, filesize(filename))

def load(filename):
    data = pickle.load(file(filename, 'r'))
    print 'Part of brain loaded from file {} ({})'.format(filename, filesize(filename))
    return data

def insert(D, key, value):
    if key in D:
        D[key].append(value)
    else:
        D[key] = [ value ]

def filetime(filename):
    return time.mktime(time.strptime(filename[filename.rfind('/')+1:filename.rfind('.wav')], '%Y_%m_%d_%H_%M_%S'))

def plot_NAP_and_energy(NAP, plt):
    plt.clf()
    plt.subplot(211)
    plt.plot(np.mean(NAP, axis=1))
    plt.xlim(xmax=len(NAP))
    plt.title('Average energy')

    plt.subplot(212)
    plt.imshow(NAP.T, aspect='auto')
    # for x in np.where(NAP > .9)[0]:
    #     plt.axvline(x, color='w')
    plt.title('NAP mean {}'.format(np.mean(NAP)))

    plt.draw()

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
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        return np.array([ [ float(r) for r in row ] for row in reader ])


def array_to_csv(filename, data, delimiter=' '):
    with open(filename, 'w') as csvfile:
        fcntl.flock(csvfile, fcntl.LOCK_EX)
        writer = csv.writer(csvfile, delimiter=delimiter)
        if len(data.shape) == 1:
            data.shape = (data.shape[0],1)
        for row in data:
            writer.writerow(row)
        fcntl.flock(csvfile, fcntl.LOCK_UN)

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

def trim_wav(sound, threshold=100):
    ''' Removes tresholded region at beginning and end '''
    right = len(sound)-1
    while sound[right] < threshold:
        right -= 1
    left = 0 
    while sound[left] < threshold:
        left += 1
    return sound[left:right]
        
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

def exact(signal, length):
    return zero_pad(signal, length)[:length]

def scale(image):
    return (image - np.min(image))/(np.max(image) - np.min(image))

def getSoundInfo(filename):
    f = open(filename[:-4]+'.txt', 'r')
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
            start,skiptime,amp,pitch,centroid = line.split(' ')
            segments.append([float(start),float(skiptime),float(amp),float(pitch),float(centroid)]) 
        if 'Sub segments (start, skiptime, amp, ' in line: enable = 1
    return startTime, totalDur, maxAmp, segments


def get_segments(filename):
    ''' Find segments in audio descriptor file'''
    _, totalDur, _, segments = getSoundInfo(filename)
    segmentTimes = []
    for item in segments:
        segmentTimes.append(item[0])    
    segmentTimes.append(totalDur) 
    #print 'utils.get_segments', segmentTimes
    return np.array(segmentTimes)

def get_amps(filename):
    _,_,_,segmentData = getSoundInfo(filename)
    return [ item[2] for item in segmentData ]

def get_most_significant_word(filename):
    amps = get_amps(filename)
    return amps.index(max(amps))
     

def getLatestMemoryWavs(howmany):
    '''
    Find the N latest recorded memory wave files. LIMITS TO 100 latest.
    '''
    path = myCsoundAudioOptions.memRecPath
    infiles = os.listdir(path)
    wavfiles = []
    for f in infiles:
        if (f[-4:] == '.wav') and ('wavALL' not in f): 
            if os.path.getsize(path+f) > 500:
                wavfiles.append(path+f)
    wavfiles.sort()
    #wavfiles = wavfiles[-100:]
    #blacklist = open('black_list.txt', 'r')
    #for line in blacklist:
    #    blackfile = find_filename.findall(line)
    #    if len(blackfile) and blackfile[0] in wavfiles:
    #        wavfiles.remove(blackfile[0])
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

def scheduler(host):
    context = zmq.Context()
    
    play_events = context.socket(zmq.PULL)
    play_events.bind('tcp://*:{}'.format(zmq_ports.SCHEDULER))
    
    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, zmq_ports.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, zmq_ports.EXTERNAL))

    projector = context.socket(zmq.PUSH)
    projector.connect('tcp://{}:{}'.format(host, zmq_ports.PROJECTOR)) 

    stateQ = context.socket(zmq.SUB)
    stateQ.connect('tcp://{}:{}'.format(host, zmq_ports.STATE))
    stateQ.setsockopt(zmq.SUBSCRIBE, b'') 
    state = stateQ.recv_json()

    poller = zmq.Poller()
    poller.register(play_events, zmq.POLLIN)
    poller.register(stateQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)

    to_be_played = []
    enable_say_something = 0

    t0 = 0
    wait_time = 0

    while True:
        events = dict(poller.poll(timeout=100))

        if stateQ in events:
            state = stateQ.recv_json()

        if state['_audioLearningStatus']:
            to_be_played = []
            wait_time = 4 ## THIS VALUE DOES NOT DO ANYTHING USEFUL, but it is ok as is that the scheduler simply sends enable_say_something when the last segment is triggered
            t0 = time.time()
            if enable_say_something: # need the local variable to avoid sending same signal several (2) times. Due to ZMQ latency?
                sender.send_json('enable_say_something 0')
                enable_say_something = 0
                
        if play_events in events:
            print 'utils scheduler disabling say something'
            sender.send_json('enable_say_something 0')
            enable_say_something = 0
            to_be_played = play_events.recv_pyobj()
            wait_time = 0

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            
            if 'clear play_events' in pushbutton and pushbutton['clear play_events']:
                print 'SCHEDULER CLEAR EVENTS'
                to_be_played = []
                sender.send_json('enable_say_something 1')
                enable_say_something = 1

        if len(to_be_played) and time.time() - t0 > wait_time:
            t0 = time.time()
            wait_time, voice1, voice2, projection, frame_size = to_be_played.pop(0)
            sender.send_json(voice1)
            sender.send_json(voice2)
            for row in np.load('{}.npy'.format(projection)):
                send_array(projector, np.resize(row, frame_size[::-1]))
                
            if len(to_be_played) == 0:
                print 'utils scheduler enabling say something'
                sender.send_json('enable_say_something 1')
                enable_say_something = 1


def true_wait(seconds):
    time.sleep(seconds)
    return True

def reboot():
    status = open('STATUS_{}'.format(time.strftime('%Y_%m_%d_%H_%M_%S')), 'w')
    call(['ps', 'aux'], stdout=status)
    call(['df', '-h'], stdout=status)
    status.close()
    call(['/etc/init.d/networking', 'stop'])
    call(['shutdown', '-r', 'now'])
    
def counter(host):
    context = zmq.Context()

    counterQ = context.socket(zmq.ROUTER)
    counterQ.bind('tcp://*:{}'.format(zmq_ports.COUNTER))

    eventQ = context.socket(zmq.SUB)
    eventQ.connect('tcp://{}:{}'.format(host, zmq_ports.EVENT))
    eventQ.setsockopt(zmq.SUBSCRIBE, b'') 

    poller = zmq.Poller()
    poller.register(counterQ, zmq.POLLIN)
    poller.register(eventQ, zmq.POLLIN)
    
    audio_ids_counter = {}
    face_ids_counter = {}

    while True:
        events = dict(poller.poll())

        if counterQ in events:
            address, _, message = counterQ.recv_multipart()
            request, value = pickle.loads(message)
            freqs = False
            if request == 'audio_id':
                audio_ids_counter[value] = audio_ids_counter[value] + 1 if value in audio_ids_counter else 1
            if request == 'face_id':
                face_ids_counter[value] = face_ids_counter[value] + 1 if value in face_ids_counter else 1
            if request == 'audio_ids_counter':
                freqs = audio_ids_counter
            if request == 'face_ids_counter':
                freqs = face_ids_counter

            counterQ.send_multipart([ address,
                                    b'',
                                    pickle.dumps(freqs) ])

        if eventQ in events:
            pushbutton = eventQ.recv_json()
            if 'save' in pushbutton:
                save('{}.{}'.format(pushbutton['save'], mp.current_process().name), [ audio_ids_counter, face_ids_counter ])

            if 'load' in pushbutton:
                audio_ids_counter, face_ids_counter = load('{}.{}'.format(pushbutton['load'], mp.current_process().name))

            
def delete_loner(counterQ, data, query, protect, deleted_ids):
    counterQ.send_pyobj([query, None])
    freqs = counterQ.recv_pyobj()
    
    histogram = np.zeros(len(data))
    for index in freqs.keys():
        histogram[index] = freqs[index]

    histogram[deleted_ids] = np.inf
    histogram[-protect:] = np.inf
    loner = np.where(histogram == min(histogram))[0][0]
    data[loner] = [[]]
    deleted_ids.append(loner)
    
    print '{} delete_id = {}'.format(query, loner)

def inside(r, q):
    (rx, ry), (rw, rh) = r
    (qx, qy), (qw, qh) = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def warm_restart():
    print 'Doing a warm restart'
    subprocess.Popen(['./warm_restart.sh'])
    
def sentinel(host):
    context = zmq.Context()
    
    life_signal_Q = context.socket(zmq.PULL)
    life_signal_Q.bind('tcp://*:{}'.format(zmq_ports.SENTINEL))

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, zmq_ports.EXTERNAL))

    poller = zmq.Poller()
    poller.register(life_signal_Q, zmq.POLLIN)

    book = {}
    save_name = False
    save_time = 0

    while True:
        events = dict(poller.poll(timeout=PROCESS_TIME_OUT*2))

        if life_signal_Q in events:
            process = life_signal_Q.recv_pyobj()
            book[process] = time.time()

        for process in book.keys():
            if not save_name and time.time() - book[process] > PROCESS_TIME_OUT*2:
                print '{} HAS DIED, SAVING'.format(process)
                save_name = brain_name()
                save_time = time.time()
                sender.send_json('save {}'.format(save_name))

        if save_name and (len(glob.glob('{}*'.format(save_name))) == NUMBER_OF_BRAINS or time.time() - save_time > SYSTEM_TIME_OUT):
            warm_restart()
                
class AliveNotifier(threading.Thread):
    def __init__(self, me, host='localhost'):
        threading.Thread.__init__(self)
        self.name = '{} PID {}'.format(me.name, me.pid)
        print self.name
        context = zmq.Context()
        self.life_signal_Q = context.socket(zmq.PUSH)
        self.life_signal_Q.connect('tcp://{}:{}'.format(host, zmq_ports.SENTINEL))

        self.start()

    def run(self):
        while true_wait(PROCESS_TIME_OUT):
            self.life_signal_Q.send_pyobj(self.name)

class SimpleLogger:
    def __init__(self, host='localhost'):
        self.out = sys.stdout
    
    def write(self, txt):
        if len(txt.rstrip()):
            self.out.write('PASSED VIA LOGGER:' + txt + '\n')
            
    def flush(self):
        self.write('DEATH')
        
class Logger:
    def __init__(self, host='localhost'):
        context = zmq.Context()
        self.logger = context.socket(zmq.PUSH)
        self.logger.connect('tcp://{}:{}'.format(host, zmq_ports.LOGGER))
    
    def write(self, txt):
        if len(txt.rstrip()):
            self.logger.send_json(txt)

    def flush(self):
        self.write('DEATH')

def log_sink():
    context = zmq.Context()
    logger = context.socket(zmq.PULL)
    logger.bind('tcp://*:{}'.format(zmq_ports.LOGGER))

    output = open('LOG_{}'.format(time.strftime('%Y_%m_%d_%H_%M_%S')), 'w')
    
    while True:
        output.write(logger.recv_json() + '\n')
        output.flush()
                        
class LoggerProcess(mp.Process):
    ''' File to write log over ZMQ socket. '''
    def run(self):
        my_logger = Logger() # Must be in run, not __init__
        sys.stdout = my_logger
        sys.stderr = my_logger

        me = mp.current_process()
        print '{} PID {}'.format(me.name, me.pid)
        AliveNotifier(me)
        
        mp.Process.run(self)
        
class MyProcess(mp.Process):
    def run(self):
        AliveNotifier(mp.current_process())
        
        mp.Process.run(self)

def brain_name():
    return 'BRAIN_{}'.format(time.strftime('%Y_%m_%d_%H_%M_%S'))
        
def find_last_valid_brain():
    files = glob.glob('BRAIN_*')
    files.sort(key=os.path.getmtime, reverse=True)
    for f in files[::NUMBER_OF_BRAINS]:
        stem = f[:f.find('.')] # We know the filename is BRAIN_XXXX.FACE RESPONDER etc
        try: 
            if len([ load(candidate) for candidate in glob.glob('{}*'.format(stem)) ]) == NUMBER_OF_BRAINS:
                return stem
        except:
            print 'Corrupt brain {}, continuing backwards'.format(stem)
    return []

def daily_routine(host):
    grind = sched.scheduler(time.time, time.sleep)
    context = zmq.Context()

    sender = context.socket(zmq.PUSH)
    sender.connect('tcp://{}:{}'.format(host, zmq_ports.EXTERNAL))

    dream_time = datetime.datetime.combine(datetime.datetime.now(), datetime.time(DREAM_HOUR))
    grind.enterabs(time.mktime(dream_time.timetuple()), 1,
                       sender.send_json, ('dream',))

    evolve_time = datetime.datetime.combine(datetime.datetime.now() + datetime.timedelta(days=1), datetime.time(EVOLVE_HOUR))
    grind.enterabs(time.mktime(evolve_time.timetuple()), 1,
                       sender.send_json, ('evolve',))

    save_time = datetime.datetime.combine(datetime.datetime.now() + datetime.timedelta(days=1), datetime.time(SAVE_HOUR))
    grind.enterabs(time.mktime(save_time.timetuple()), 1,
                       sender.send_json, ('save',))
    
    reboot_time = datetime.datetime.combine(datetime.datetime.now() + datetime.timedelta(days=1), datetime.time(REBOOT_HOUR))
    grind.enterabs(time.mktime(reboot_time.timetuple()), 1,
                       sender.send_json, ('reboot',))

    grind.run()
        
def load_esn(filename):
    import Oger
    import mdp
    numpies = [ np.load('{}_{}.npy'.format(filename, i)) for i in range(6) ]

    _reservoir, _linear = json.load(open(filename+'.npy','r'))
    _reservoir['_dtype'] = np.dtype('float64')
    _reservoir['nonlin_func'] = Oger.utils.TanhFunction
    _reservoir['initial_state'] = 0
    _reservoir['states'] = numpies[0]
    _reservoir['w'] = numpies[1]
    _reservoir['w_in'] = numpies[2]
    _reservoir['w_bias'] = numpies[3]
        
    reservoir = Oger.nodes.LeakyReservoirNode(leak_rate=.0)
    reservoir.__dict__ = _reservoir

    linear = readout = mdp.nodes.LinearRegressionNode()

    _linear['_dtype'] = np.dtype('float64')
    _linear['_xTx'] = numpies[4]
    _linear['_xTy'] = numpies[5]
    
    linear.__dict__ = _linear

    flow = mdp.hinet.FlowNode(reservoir + linear)
    flow._train_phase_started = True
    
    return flow
        
def dump_esn(net, filename):
    ''' Stores what does not go into JSON as numpy arrays. Much faster than pickle. '''
    reservoir = net[0]
    linear = net[1]

    numpies = []
    _reservoir = reservoir.__dict__.copy()
    del _reservoir['_dtype']
    del _reservoir['initial_state']
    del _reservoir['nonlin_func']
    numpies.append(reservoir.states)
    del _reservoir['states']
    numpies.append(reservoir.w)
    del _reservoir['w']
    numpies.append(reservoir.w_in)
    del _reservoir['w_in']
    numpies.append(reservoir.w_bias)
    del _reservoir['w_bias']

    _linear = linear.__dict__.copy()
    del _linear['_dtype']
    numpies.append(linear._xTx)
    del _linear['_xTx']
    numpies.append(linear._xTy)
    del _linear['_xTy']

    json.dump([_reservoir, _linear], open(filename+'.npy','w')) # Ha! .npy for fun. When massive restart, you can remove this.

    for i,N in enumerate(numpies):
        np.save('{}_{}'.format(filename, i), N)
