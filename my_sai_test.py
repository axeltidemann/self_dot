import time
import glob
import itertools
import multiprocessing as mp
import cPickle as pickle

import numpy as np
from scipy.io import wavfile
from scipy.cluster.vq import kmeans, vq
import cv2

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
    
    while window_height <= height:
        end_row = window_height

        while end_row <= height:

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

def sai_sparse_codes(sai_video_marginals, k):
    all_marginals = list(itertools.chain.from_iterable(sai_video_marginals))
    codebooks = sai_codebooks(all_marginals, k)
    return [ sai_histogram(s, codebooks, k) for s in sai_video_marginals ]

def _valid_file(filename, threshold=.1):
    try:
        return utils.get_segments(filename)[-1] > threshold
    except:
        return False    

def _cochlear_trim_sai_marginals(filename_and_indexes):
    try:
        filename, norm_segstart, norm_segend, audio_id, NAP_detail = filename_and_indexes

        if NAP_detail == 'high':
            try: 
                NAP = utils.csv_to_array(filename+'cochlear'+NAP_detail)
            except:
                NAP = brain.cochlear(filename, stride=1, rate=44100, apply_filter=0, suffix='cochlear'+NAP_detail)
        if NAP_detail == 'low':
            try: 
                NAP = utils.csv_to_array(filename+'cochlear'+NAP_detail)
            except: 
                NAP = brain.cochlear(filename, stride=IO.NAP_STRIDE, rate=IO.NAP_RATE, apply_filter=0, suffix='cochlear'+NAP_detail) # Seems to work best, in particular when they are all the same.

        num_channels = NAP.shape[1]
        input_segment_width = 2048
        sai_params = CreateSAIParams(num_channels=num_channels,
                                     input_segment_width=input_segment_width,
                                     trigger_window_width=input_segment_width,
                                     sai_width=1024)

        sai = pysai.SAI(sai_params)

        NAP = utils.trim_right(NAP[ np.int(np.rint(NAP.shape[0]*norm_segstart)) : np.int(np.rint(NAP.shape[0]*norm_segend)) ], threshold=.05)
        sai_video = [ np.copy(sai.RunSegment(input_segment.T)) for input_segment in utils.chunks(NAP, input_segment_width) ]
        return [ [ filename, audio_id, [ sai_rectangles(frame) for frame in sai_video ]] ]

    except:
        print utils.print_exception()
        return [[filename, audio_id, None]]

def experiment(filenames, k):
    t0 = time.time()
    pool = mp.Pool() 
    sai_video_marginals = pool.map(_cochlear_trim_sai_marginals, filenames)
    sai_video_marginals = list(itertools.chain.from_iterable(sai_video_marginals))
    pool.close()
    t1 = time.time()
    print 'Cochlear SAI marginals calculated in {} seconds'.format(t1 - t0)
    sparse_codes = sai_sparse_codes([ marginals for _,_,marginals in sai_video_marginals if marginals is not None ], k)
    print 'Sparse codes calculated in {} seconds'.format(time.time() - t1)
    return sparse_codes
    
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sparse_codes = experiment([ filename for filename in glob.glob('testing/*wav') if _valid_file(filename) ], k=4)
    
    #pickle.dump(sai_video_marginals, open('SAIVIDEOMARGINALS', 'w'))    
    plt.ion()
    plt.matshow(sparse_codes, aspect='auto')
    plt.colorbar()
