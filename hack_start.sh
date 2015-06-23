#!/bin/bash

# This is the startup file used for ACM Creativity + Cognition in Glasgow, 2015. 
# MacBook Pro running 10.7.5 was used.

python video.py &> video_log &
python self_dot.py &

sleep 10

python communication.py calibrateAudio

sleep 10

python communication.py display2 1
#python communication.py fullscreen 1
python communication.py memoryRecording 1
python communication.py selfDucking 1
python communication.py autolearn 1
python communication.py ambientSound 1
python communication.py autorespond_sentence 1
python communication.py roboActive 1
