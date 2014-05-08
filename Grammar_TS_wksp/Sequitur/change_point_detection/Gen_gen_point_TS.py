'''
Created on 6 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

def gen_piecewise_latent(mus, sigmas, n_points, intensity, weights):
    n_change_points = np.random.poisson(intensity * n_points)
    change_points = list(sorted(set(np.asarray(np.random.randint(0, n_points, n_change_points)))))
    generated = []
    symbols = []
    j = np.random.choice(len(mus))
    symbols.append(j)
    generated.extend(list(np.random.multivariate_normal(mus[j], sigmas[j], change_points[0])))
    for i in xrange(len(change_points) - 1):
        j = np.random.choice(len(mus))
        symbols.append(j)
        generated.extend(list(np.random.multivariate_normal(mus[j], sigmas[j], change_points[i+1] - change_points[i])))
    symbols.append(j)
    j = np.random.choice(len(mus))
    generated.extend(list(np.random.multivariate_normal(mus[j], sigmas[j], n_points - change_points[-1])))
    return symbols, change_points, range(n_points), np.vstack(generated)

n_symbols = 3
    
mus = [[0.0, 0.0], [-2.0, 2.0], [-1.0, 1.0]]
sigmas = [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]

mus = [np.asanyarray(x, dtype = np.double) for x in mus]
sigmas = [np.asanyarray(x, dtype = np.double) for x in sigmas]

weights = np.random.power(2, n_symbols)

symbols, change_points, x_axis, values = gen_piecewise_latent(mus, sigmas, 1000, 0.02, weights)

def EMA(TS, alpha):
    result = np.zeros(len(TS))
    result[0] = TS[0]
    for i in xrange(1, len(TS)):
        result[i] = alpha * TS[i] + (1 - alpha) * result[i-1]
    return result

EMA_first_dim = EMA(values[:, 0], 0.1)
EMA_second_dim = EMA(values[:, 1], 0.1)

print len(x_axis)
print len(values)
print len(symbols)
print len(change_points)

colors = ['yellow', 'magenta', 'cyan']

for i, symbol in enumerate(symbols[:-1]):
    plt.vlines(change_points[i-1], -10, 10, linewidth = 5, color = colors[symbol])
plt.plot(x_axis, values, linestyle = 'None', marker = '.')
plt.plot(x_axis, EMA_first_dim, color = 'k')
plt.plot(x_axis, EMA_second_dim, color = 'k')
plt.title('Bivariate normals')
plt.ylabel('Measurements')
plt.xlabel('Time')
plt.legend(('First of data', 'Second dim of data', 'First EMA', 'Second EMA'))
plt.savefig('Example_of_TS.png', dpi = 300)
plt.close()