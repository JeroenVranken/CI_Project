#!/usr/bin/python
# Run unix date command 3 times 
import os
import time
for x in range(0,3):
	os.system("torcs -r /home/petra/Documents/computational_intelligence/CI_Project/torcs-client/quickrace.xml")
	time.sleep(0.5)