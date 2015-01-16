#!/bin/bash          

sleep 5
qjackctl -s &

cd /home/self/Projects/self_dot/

now=$(date +"%Y_%m_%d_%H_%M_%S")

python -u self_dot.py &> "OUTPUT_$now" &

gnome-terminal --window-with-profile=self_output -e "tail -f `ls -t OUTPUT* | head -1`" &

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
python communication.py load

xte 'mousemove 10 10'

wmctrl -a Output
