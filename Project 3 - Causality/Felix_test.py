import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


textfile = 'data_237.csv'

df = np.genfromtxt(textfile,delimiter=',')

fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
ax1.hist(df[:,1],bins=10)
ax2.hist(df[df[:,1]==0,2],bins=10)
ax3.hist(df[df[:,1]==0,3],bins=10)
ax4.hist(df[df[:,1]==0,4],bins=10)
#plt.legend(['Q','R','S','T'])
plt.show()