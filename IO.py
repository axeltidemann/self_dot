import os

import cv2
import numpy as np
import myCsoundAudioOptions

def video(sense, camera, projector):
    print 'VIDEO PID', os.getpid()
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    video_feed = cv2.VideoCapture(0)
    frame_size = (160, 90)
    
    while True:
        _, frame = video_feed.read()
        frame = cv2.resize(frame, frame_size)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if sense.value:
            camera.append(np.ndarray.flatten(gray_image)/255.)

        try:
            cv2.imshow('Output', cv2.resize(np.reshape(projector.popleft(),
                                                       frame_size[::-1]), (640,360)))
        except:
            cv2.imshow('Output', np.random.rand(360,640))

        cv2.waitKey(100)

def audio(sense, mic, speaker):
    print 'AUDIO PID', os.getpid()
    import csnd6
    cs = csnd6.Csound()
    arguments = csnd6.CsoundArgVList()
    arguments.Append("dummy")
    arguments.Append("self_dot.csd")
    csoundCommandline = myCsoundAudioOptions.myAudioDevices
    comlineParmsList = csoundCommandline.split(' ')
    for item in comlineParmsList:
        arguments.Append("%s"%item)
    cs.Compile(arguments.argc(), arguments.argv())
    stopflag = 0
    #fft_audio_in1 = np.zeros(1024)
    #fft_audio_in2 = np.zeros(1024)
    offset = 0

    while not stopflag:
        stopflag = cs.PerformKsmps()

        offset += 0.01
        offset %= 200
        cs.SetChannel("freq_offset", offset)
        #test1 = cs.GetPvsChannel(fft_audio_in1, 0)
        #test2 = cs.GetPvsChannel(fft_audio_in2, 1)

        # get Csound channel data
        #audioStatus = cs.GetChannel("audioStatus")
        if sense.value:
            mic.append([ cs.GetChannel("level1"), cs.GetChannel("envelope1"),
                         cs.GetChannel("pitch1"), cs.GetChannel("centroid1") ])
        
        try:
            sound = speaker.popleft()
            cs.SetChannel("respondLevel1", sound[0])
            cs.SetChannel("respondEnvelope1", sound[1])
            cs.SetChannel("respondPitch1", sound[2])
            cs.SetChannel("respondCentroid1", sound[3])
        except:
            cs.SetChannel("respondLevel1", 0)
            cs.SetChannel("respondEnvelope1", 0)
            cs.SetChannel("respondPitch1", 0)
            cs.SetChannel("respondCentroid1", 0)

