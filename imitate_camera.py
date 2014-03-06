import multiprocessing

import numpy as np
import cv2

def see(eye_q, memorize_q):
    cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
    camera = cv2.VideoCapture(0)

    i = 0
    while True:
        _, frame = camera.read()
        frame = cv2.resize(frame, (640,360))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Input", gray_image)    
        eye_q.put(np.ndarray.flatten(gray_image)/255.)
        cv2.waitKey(10)

        i += 1
        if i == 100:
            memorize_q.put(np.ndarray.flatten(gray_image)/255.)
            i = 0
    
def learn(memorize_q, brain_q):
    # Interestingly enough, importing Oger in the main file causes
    # troubles due to the parallelism. But if imported only in this
    # process, we are right as rain.
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

def display(eye_q, brain_q):
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

    neural_net = brain_q.get()

    while True:
        input_photo = eye_q.get()
        input_photo.shape = (1, input_photo.shape[0])

        try:
            neural_net = brain_q.get_nowait()
        except:
            pass

        cv2.imshow("Output", np.reshape(neural_net(input_photo), (360,640)))           

def buttons():
    import Tkinter # Same as with Oger - fine to import it here, disaster elsewhere.
    top = Tkinter.Tk()
    
    def memorize():
        print 'Remember the time!'

    def forget():
        print 'Forget the time!'

    Tkinter.Button(top, text='Memorize', command=memorize).pack()
    Tkinter.Button(top, text='Forget', command=forget).pack()
    top.mainloop()

if __name__ == "__main__":
    eye_q = multiprocessing.Queue()
    brain_q = multiprocessing.Queue()
    memorize_q = multiprocessing.Queue()
    
    multiprocessing.Process(target=see, args=(eye_q, memorize_q)).start()
    multiprocessing.Process(target=learn, args=(memorize_q, brain_q)).start()
    multiprocessing.Process(target=display, args=(eye_q, brain_q)).start()
    multiprocessing.Process(target=buttons).start()

    cv2.destroyAllWindows()
