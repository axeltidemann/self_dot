import multiprocessing

import numpy as np
import mdp
from reservoir_nodes import LeakyReservoirNode
import cv2

def see(photo_q, learn_q):
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    camera = cv2.VideoCapture(0)

    i = 0
    while True:
        _, frame = camera.read()
        frame = cv2.resize(frame, (640,360))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Input", gray_image)    
        photo_q.put(np.ndarray.flatten(gray_image)/255.)
        cv2.waitKey(10)

        i += 1
        if i == 100:
            learn_q.put(np.ndarray.flatten(gray_image)/255.)
            i = 0
    
def learn(learn_q, brain_q):
    photos = []

    while True:
        photos.append(learn_q.get())

        if len(photos) < 2:
            continue

        if len(photos) > 20:
            photos.pop(np.random.randint(len(photos)-1))
            print '>20 examples. Removing at random.'

        reservoir = LeakyReservoirNode(output_dim=100, leak_rate=0.8, bias_scaling=.2) 
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        neural_net = mdp.hinet.FlowNode(reservoir + readout)

        x = np.array(photos[:-1])
        y = np.array(photos[1:])

        neural_net.train(x,y)
        neural_net.stop_training()

        brain_q.put(neural_net)

        print 'Learning phase completed, using ', len(photos), 'photos.'

def display(photo_q, brain_q):
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

    neural_net = False

    while True:
        input_photo = photo_q.get()
        input_photo.shape = (1, input_photo.shape[0])

        try:
            neural_net = brain_q.get_nowait()
        except:
            pass

        if not neural_net:
            output = np.random.rand(360, 640)
        else:
            output = np.reshape(neural_net(input_photo), (360,640))

        cv2.imshow("Output", output)            

if __name__ == "__main__":
    photo_q = multiprocessing.Queue()
    brain_q = multiprocessing.Queue()
    learn_q = multiprocessing.Queue()
    
    multiprocessing.Process(target=see, args=(photo_q, learn_q)).start()
    multiprocessing.Process(target=learn, args=(learn_q, brain_q)).start()
    p = multiprocessing.Process(target=display, args=(photo_q, brain_q))
    p.start()
    p.join()
    
    cv2.destroyAllWindows()
