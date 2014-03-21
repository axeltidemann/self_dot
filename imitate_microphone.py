# Tries to predict next sound parameters using reservoir
# computing (e.g. Echo State Networks). Input, output and learning in
# sepearate processes.
#
# Author: Axel Tidemann, axel.tidemann@gmail.com

import multiprocessing

import csnd6
import time
import numpy as np
import cv2

def hear(ear_q, memorize_q):
    # This should be changed to use microphone input instead
    
    cs = csnd6.Csound()
    cs.Compile("self_audio_csnd_test.csd")
    cs.Start()
    stopflag = 0
    #fft_audio_in1 = numpy.zeros(1024)
    #fft_audio_in2 = numpy.zeros(1024)
    offset = 0

    level1 = []
    envelope1 = []
    pitch1 = []
    centr1 = []

    while not stopflag:
        stopflag = cs.PerformKsmps()

        offset += 0.01
        offset %= 200
        cs.SetChannel("freq_offset", offset)
        #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
        #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

        # get Csound channel data
        level1.append(cs.GetChannel("level1"))
        envelope1.append(cs.GetChannel("envelope1"))
        pitch1.append(cs.GetChannel("pitch1"))
        centr1.append(cs.GetChannel("centroid1"))

        # self does its imitation magic and writes new values to Csound channels
        # for now just copying the input values to test the signal chain

    

    
    cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
    camera = cv2.VideoCapture(0)

    i = 0
    while True:
        _, frame = camera.read()
        frame = cv2.resize(frame, (320,180))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Input', gray_image)    
        ear_q.put(np.ndarray.flatten(gray_image)/255.)

        i += 1
        if i == 100:
            memorize_q.put(np.ndarray.flatten(gray_image)/255.)
            i = 0

        cv2.waitKey(10)
    
def learn(memorize_q, brain_q):
    # Interestingly enough, importing Oger at the top of this file
    # causes troubles due to the parallelism.
    import Oger
    import mdp

    photos = []

    while True:
        photos.append(memorize_q.get())

        if len(photos) < 2:
            continue

        if len(photos) > 20:
            photos.pop(np.random.randint(len(photos)-1))
            print '>20 examples. Removing at random.'

        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=100, leak_rate=0.8, bias_scaling=.2) 
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        neural_net = mdp.hinet.FlowNode(reservoir + readout)

        x = np.array(photos[:-1])
        y = np.array(photos[1:])

        neural_net.train(x,y)
        neural_net.stop_training()

        brain_q.put(neural_net)

        print 'Learning phase completed, using', len(photos), 'photos.'

def play(ear_q, brain_q):
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

    neural_net = brain_q.get()

    while True:
        
        # set Csound channel data
        cs.SetChannel("imitateLevel1", level1[-1])
        cs.SetChannel("imitateEnvelope1", envelope1[-1])
        cs.SetChannel("imitatePitch1", pitch1[-1])
        cs.SetChannel("imitateCentroid1", centr1[-1])



        
        input_photo = ear_q.get()
        input_photo.shape = (1, input_photo.shape[0])

        try:
            neural_net = brain_q.get_nowait()
        except:
            pass

        cv2.imshow('Output', np.reshape(neural_net(input_photo), (180,320)))
        cv2.waitKey(1) # Needed for Windows compatability?
    
if __name__ == '__main__':
    ear_q = multiprocessing.Queue()
    brain_q = multiprocessing.Queue()
    memorize_q = multiprocessing.Queue()
        
    multiprocessing.Process(target=hear, args=(ear_q, memorize_q)).start()
    multiprocessing.Process(target=learn, args=(memorize_q, brain_q)).start()
    multiprocessing.Process(target=play, args=(ear_q, brain_q)).start()

    try:
        raw_input('')
    except KeyboardInterrupt:
        map(lambda x: x.terminate(), multiprocessing.active_children())


    

