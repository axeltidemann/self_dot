#!/bin/bash          

killall -9 python
sleep 10
sudo ./usbreset `lsusb | grep Webcam | awk '{printf "/dev/bus/usb/%s/%s", $2, $4}' | sed -e 's/://'`
./startup.sh
