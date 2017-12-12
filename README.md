# Computational Intelligence Project

## Code location
All final commented code is in the ```clean_code``` folder.

### Running the driver
To start a driver, first select a quickrace in torcs, and select driver **scr-server_1**. Start the race, and then run:
```
./start.sh
```
from a terminal.

### Training a network
To train a model, specify which model to train in ```train_simpleNet.py```. All networks are specified in ```networks.py```, which can be imported in ```train_simpleNet.py``` by using:
```
from networks import NETWORKNAME
```

Training is started by simply calling:
```
train_simpleNet.py
```
After each epoch, the weights are saves in: ```saves/networkname```.

### NEAT
The code for the neat driver is in the neat-driver.py it uses the python-neat package. 
All code of this package is in the 'neat' folder (obtained from the pyton-neat github).
Furthermore neat-driver.py uses the config-neat and winner-feedworward file.

### Deep reinforcement learning
The code for the reinforcement learning is in the drl.py. It is also necessary to start torcs to start training and run the start.sh file in a terminal. The code will automatically restart the game when necessary. 
