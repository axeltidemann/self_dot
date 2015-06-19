#!/usr/bin/python
# -*- coding: latin-1 -*-

#    Copyright 2014 Oeyvind Brandtsegg and Axel Tidemann
#
#    This file is part of [self.]
#
#    [self.] is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 
#    as published by the Free Software Foundation.
#
#    [self.] is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with [self.].  If not, see <http://www.gnu.org/licenses/>.

''' [self.]

@author: Axel Tidemann, Øyvind Brandtsegg
@contact: axel.tidemann@gmail.com, obrandts@gmail.com
@license: GPL
'''

# A small program that does face recognition (quite poorly, I'm afraid). This was part of a MindStorms robot I built, which
# moved its head to follow persons in the room. It works standalone as well. This will be removed in the future, it is just here
# for a starting point to use the camera.
#
# Author: Axel Tidemann, axel.tidemann@gmail.com

import thread
import sys

import numpy as np
import ipdb
import Oger
import mdp
import cv2
 
HAAR_CASCADE_PATH = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
#HAAR_CASCADE_PATH = "C:/zip/2011_laptop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"

frame = []
faces = []
names = []
flow = []
finished_training = False
training_set = []

def detect():
    global faces
    global frame

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    camera = cv2.VideoCapture(0)
    storage = cv2.cv.CreateMemStorage()
    cascade = cv2.cv.Load(HAAR_CASCADE_PATH)
  
    i = 0
    while True:
        rval, frame = camera.read()

        frame = cv2.resize(frame, (640, 360)) # 16:9 aspect ratio of Logitech USB camera
 
        # Only run the Detection algorithm every 5 frames to improve performance
        if i%5==0:
            faces = [ (x,y,w,h) for (x,y,w,h),n in cv2.cv.HaarDetectObjects(cv2.cv.fromarray(frame), cascade, storage, 1.2, 2, cv2.cv.CV_HAAR_DO_CANNY_PRUNING, (50,50)) ]

        for (x,y,w,h) in faces: 
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)

        if faces and finished_training:
            cv2.putText(frame, names[np.argmax(flow(np.atleast_2d(mugshot())))], (x,y),
                        cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness = 2)

        cv2.imshow("Video", frame)
        i += 1
        
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()

def mugshot():
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x,y,w,h = faces[0] # Assuming we only want first face.
    cropped_image = gray_image[ y:y+h, x:x+w ]
    scaled_image = cv2.resize(cropped_image, (100,100))
    return scaled_image.flatten()/255.

def get_name():
    global training_set
    global flow
    global names
    global finished_training

    num_pics = 20

    while True:
        names.append(raw_input('Enter name of ugly mug: '))
        finished_training = False
        print 'Taking picture',
        i = 0 
        while i < num_pics:
            try:
                training_set.append(mugshot())
                print i,
                sys.stdout.flush()
                cv2.waitKey(200)
                i += 1
            except:
                pass
            
        reservoir = Oger.nodes.ReservoirNode(output_dim = 50,
                                             spectral_radius = 0)
        #readout = Oger.nodes.RidgeRegressionNode(ridge_param = 0.0001)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=True)
        flow = mdp.hinet.FlowNode(reservoir + readout)
        target = np.repeat(np.eye(len(names)), (np.ones(len(names))*num_pics).astype(int), axis=0)
        flow.train(np.array(training_set), target)
        flow.stop_training()
        finished_training = True
        print ', '.join(names), 'stored in ESN'
        print 'RMSE', Oger.utils.rmse(flow(np.array(training_set)), target)

def recognize():
    thread.start_new_thread(get_name, ())
    #waitKey and imshow need to be in main thread
    detect()

if __name__ == "__main__":
    recognize()
