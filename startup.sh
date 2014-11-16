#!/bin/bash          

#startup JACK
qjackctl -s &

python self_dot.py
