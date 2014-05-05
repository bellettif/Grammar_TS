'''
Created on 5 avr. 2014

@author: francois
'''

from matplotlib import pyplot as plt
import numpy as np

import data.symbolic

print data.symbolic.achu_set.keys()

target_ts = data.symbolic.achu_set[4]

plt.plot(target_ts, linestyle = 'None', marker = 'o')
plt.show()

