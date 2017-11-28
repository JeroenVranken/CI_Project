#!/usr/bin/python3
# Run unix date command 3 times 
import os

for x in range(0,1):
	os.system("./start_torcs.sh")

os.system("./dubble_check_start.sh")