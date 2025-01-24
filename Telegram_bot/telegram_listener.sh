#!/bin/bash
sleep 39
export DISPLAY=:0
tilix -e /home/pi/miniforge3/envs/quantext/bin/python3.12 /home/pi/Scripts/tel_bot_listener.py