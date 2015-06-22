#!/bin/bash

python video.py &> video_log &
python robocontrol.py &> robocontrol_log &
python self_dot.py &

sleep 10

python communication.py calibrateAudio

sleep 10

python communication.py display2 1
python communication.py fullscreen 1
python communication.py memoryRecording 1
python communication.py selfDucking 1
python communication.py autolearn 1
python communication.py ambientSound 1
python communication.py autorespond_sentence 1
python communication.py roboActive 1
