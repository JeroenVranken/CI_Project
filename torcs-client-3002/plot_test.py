import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

all_losses = [1,2,3,4,5]

filename= 'aaatest'

plt.figure()
plt.plot(all_losses)
plt.savefig("losses_" + filename + ".png")