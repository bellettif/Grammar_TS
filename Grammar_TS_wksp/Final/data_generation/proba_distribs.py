'''
Created on 30 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

symbols = ['a', 'b', 'c']
p_bar_weights = np.asarray([3.0, 2.0, 1.0])
p_bar_weights /= np.sum(p_bar_weights)

p_wildcard = p_bar_weights
p_rule = 0.7

p_combined = p_rule * p_bar_weights + (1.0 - p_rule) * p_wildcard

orange_color = '#FF9933'
blue_color = '#3366FF'

width = 0.8
x_pos = np.arange(1, 4)

plt.bar(x_pos, 
        height = p_bar_weights,
        width = width,
        color = blue_color)
plt.title('P without noise')
plt.ylabel('P bar distrib')
plt.xticks(x_pos + 0.5 * width, symbols)
plt.savefig('P_without_noise.png', dpi = 300)
plt.close()

plt.bar(x_pos, 
        height = p_wildcard,
        width = width,
        color = orange_color)
plt.title('P of wildcard')
plt.ylabel('P wildcard distrib')
plt.xticks(x_pos + 0.5 * width, symbols)
plt.savefig('P_wildcard.png', dpi = 300)
plt.close()

plt.bar(x_pos,
        height = p_rule * p_bar_weights,
        width = width,
        color = blue_color)
plt.bar(x_pos,
        height = (1.0 - p_rule) * p_wildcard,
        width = width,
        bottom = p_rule * p_bar_weights,
        color = orange_color)
plt.title('P combined (noisy sequence)')
plt.ylabel('P distrib')
plt.xticks(x_pos + 0.5 * width, symbols)
plt.legend(('Rules', 'Noise'), 'upper right')
plt.savefig('P_combined.png', dpi = 300)
plt.close()


