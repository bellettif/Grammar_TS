'''
Created on 21 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

x_values = np.linspace(-10, 10, 1000, True)
y_values = np.sin(x_values)
y_values += np.random.normal(0, 0.01, len(x_values))

#plt.plot(x_values, y_values, linestyle = 'None', marker = '.')
#plt.show()

diff_y_values = np.diff(y_values)
plt.plot(x_values[:-1], diff_y_values, linestyle = 'None', marker = '.')
plt.show()