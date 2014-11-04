import time
import glob
import itertools

import numpy as np
from scipy.io import wavfile
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

import sai as pysai
import IO
import utils
import brain

def CreateSAIParams(sai_width, num_triggers_per_frame=2, **kwargs):
  """Fills an SAIParams object using reasonable defaults for some fields."""
  return pysai.SAIParams(sai_width=sai_width,
                         # Half of the SAI should come from the future.
                         future_lags=sai_width / 2,
                         num_triggers_per_frame=num_triggers_per_frame,
                         **kwargs)

def sai_rectangles(sai_frame, channels=32, lags=16):
    center = sai_frame.shape[1]/2
    height = sai_frame.shape[0]
    width = sai_frame.shape[1]

    window_height = channels
    window_width = lags

    marginal_values = []
    
    while window_height < height:
        end_row = window_height

        while end_row < height:

            while window_width <= width:
                #print 'SAI frame section {} {} {} {}'.format(end_row - window_height, end_row, center - (window_width - width/2) if window_width > width/2 else center, np.clip(center + window_width, center, width))
                r = sai_frame[end_row - window_height:end_row, center - (window_width - width/2) if window_width > width/2 else center:np.clip(center + window_width, center, width)]
                r_resized = cv2.resize(r, (lags, channels))
                marginal_values.append(np.hstack((np.mean(r_resized, axis=0), np.mean(r_resized, axis=1))))
                window_width *= 2

            end_row += window_height/2        
            window_width = lags

        window_height *= 2

    return marginal_values

def sai_codebooks(marginals, k):
    code_books = []
    for rectangle in zip(*marginals):
        obs = np.vstack(rectangle)
        #print '{} observations, {} features, k={}'.format(obs.shape[0], obs.shape[1], k)
        code_books.append(kmeans(obs, k)[0])

    return code_books

def sai_histogram(sai_video_marginals, codebooks, k):
    sparse_code = np.zeros(len(codebooks)*k)
    for marginals in sai_video_marginals:
        sparse_frame = [ np.zeros(k) ] * len(codebooks)

        for rect, code, frame in zip(marginals, codebooks, sparse_frame):
            frame[vq(np.atleast_2d(rect), code)[0]] = 1 

        sparse_code += np.hstack(sparse_frame) 
    return sparse_code

def sai_sparse_codes(sai_videos, k):
    sai_video_marginals = [[sai_rectangles(sai_frame) for sai_frame in video] for video in sai_videos]
    all_marginals = list(itertools.chain.from_iterable(sai_video_marginals))
    codebooks = sai_codebooks(all_marginals, k)
    return [ sai_histogram(s, codebooks, k) for s in sai_video_marginals ]
    

if __name__ == '__main__':

    t0 = time.time()
    
    NAPs = [ utils.trim_right(brain.cochlear(filename, stride=1, new_rate=44100, apply_filter=0), threshold=.05) for filename in glob.glob('memory_recordings/*wav') if utils.get_segments(filename)[-1] > .1 ]
    t1 = time.time()
    print 'Cochlear calculated in {} seconds'.format(t1 - t0)

    num_channels = NAPs[0].shape[1]
    input_segment_width = 2048
    sai_params = CreateSAIParams(num_channels=num_channels,
                                 input_segment_width=input_segment_width,
                                 trigger_window_width=input_segment_width,
                                 sai_width=1024)

    sai = pysai.SAI(sai_params)

    sai_videos = []
    for NAP in NAPs:
        sai_videos.append([ np.copy(sai.RunSegment(input_segment.T)) for input_segment in utils.chunks(NAP, input_segment_width) ] ) # NOTICE THE NP.COPY FOR JESUS HOLY MOTHER OF JOSEPH SAKE!!! HOLY CRAP!
        sai.Reset()

    print 'SAI video calculated in {} seconds'.format(time.time() - t1)

    k = 256
    sparse_codes = sai_sparse_codes(sai_videos, k)

    plt.ion()
    plt.matshow(sparse_codes, aspect='auto')
    
