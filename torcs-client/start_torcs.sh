#!/usr/bin/python3
# Run unix date command 3 times 
import os
import numpy as np

com = "torcs -r ~/Documents/computational_intelligence/CI_Project/torcs-client/quickrace_"
tracks = ["forza.xml", "alpine2.xml", "corkscrew.xml"]
com = com + tracks[np.random.randint(0, len(tracks))]

for x in range(0,1):
	os.system(com)

os.system("./start_torcs.sh")