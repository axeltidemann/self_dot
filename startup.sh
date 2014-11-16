#!/bin/bash          

sleep 5
qjackctl -s &

cd /home/self/Projects/self_dot/
python self_dot.py > OUTPUT 2>&1 &

sleep 10

python communication.py calibrateAudio

sleep 10

python communication.py display2 1
python communication.py fullscreen 1
python communication.py memoryRecording 1
python communication.py selfDucking 1
python communication.py autolearn 1
python communication.py autorespond_sentence 1
python communication.py roboActive 1
