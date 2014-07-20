'''
Created on 20 juil. 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from plot_convention.colors import b_orange, b_blue

n_symbols = range(6,13)

achu_lks = [-46.52,
            -45.10,
            -44.21, 
            -42.85,
            -42.21, 
            -41.55,
            -41.29]

oldo_lks = [-45.23,
            -42.36,
            -42.29,
            -41.98,
            -41.41,
            -40.55,
            -40.51]

plt.plot(n_symbols, achu_lks, 
         c = b_orange, marker = 'o')
plt.plot(n_symbols, oldo_lks, 
         c = b_blue, marker = 'o')
plt.title('Average likelihood of 20 best estimates (over 100)')
plt.xlabel('Number of non-terminal symbols')
plt.ylabel('Avg log lk over samples')
plt.legend(('Achu', 'Oldo'), 'upper left')
plt.savefig('Final_plot.png', dpi = 300)
plt.close()