'''
Created on 19 juin 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt

from load_data import achu_file_contents, \
                        oldo_file_contents, \
                        no_rep_achu_file_contents, \
                        no_rep_oldo_file_contents, \
                        no_g_achu_file_contents, \
                        no_g_no_rep_achu_file_contents, \
                        no_g_oldo_file_contents, \
                        no_g_no_rep_oldo_file_contents, \
                        achu_file_names, \
                        oldo_file_names

all_symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

symbol_colors = ['lightblue',
                 'lightcoral',
                 'lightcyan',
                 'lightgoldenrodyellow',
                 'lightgreen',
                 'lightsalmon',
                 'lightseagreen']

print 'Raw data'
print [len(x.split(' ')) for x in achu_file_contents.values()]
print [len(x.split(' ')) for x in oldo_file_contents.values()]
print''
print 'No repetitions of symbol a'
print [len(x.split(' ')) for x in no_rep_achu_file_contents.values()]
print [len(x.split(' ')) for x in no_rep_oldo_file_contents.values()]
print ''
print 'No g symbol:'
print [len(x.split(' ')) for x in no_g_achu_file_contents.values()]
print [len(x.split(' ')) for x in no_g_oldo_file_contents.values()]
print ''
print 'No g symbol nor repetitions of symbol a'
print [len(x.split(' ')) for x in no_g_no_rep_achu_file_contents.values()]
print [len(x.split(' ')) for x in no_g_no_rep_oldo_file_contents.values()]

maximum_length = max(max([len(x.split(' ')) for x in achu_file_contents.values()]),
                     max([len(x.split(' ')) for x in oldo_file_contents.values()]))

def print_lengths(data_set,
                  data_set_name, 
                  file_names):
    data_set_values = [data_set[x] for x in file_names]
    plt.title(data_set_name + ' lengths')
    plt.ylabel('Length of sequences')
    plt.ylim(0, maximum_length)
    width = 0.4
    x_pos = width * 0.5 + np.arange(len(data_set))
    plt.bar(x_pos, [len(x.split(' ')) for x in data_set_values], width,
            color = 'powderblue')
    plt.xticks(x_pos + 0.5 * width, 
               file_names,
               rotation = 'vertical',
               fontsize = 4)
    plt.savefig(data_set_name + ' lengths.png', dpi = 600)
    plt.close()
    
def print_symbol_counts(data_set, 
                             data_set_name,
                             file_names):
    data_set_values = [data_set[x].split(' ') for x in file_names]
    counts = np.zeros((len(all_symbols), (len(data_set))))
    for i_sym, symbol in enumerate(all_symbols):
        for i_seq, seq in enumerate(data_set_values):
            count = len(filter(lambda x : x == symbol,
                               seq))
            counts[i_sym, i_seq] = count
    width = 0.4
    x_pos = width * 0.5 + np.arange(len(data_set))
    plt.title(data_set_name + ' symbol counts')
    plt.ylabel('Count of each symbol')
    cum_sum = np.cumsum(counts, axis = 0)
    plt.ylim((0, np.max(cum_sum) * 1.3))
    for i_sym in xrange(counts.shape[0]):
        if i_sym == 0:
            plt.bar(x_pos,
                    height = counts[i_sym],
                    width = width,
                    color = symbol_colors[i_sym])
        else:
            plt.bar(x_pos,
                    height = counts[i_sym],
                    width = width,
                    bottom = cum_sum[i_sym - 1],
                    color = symbol_colors[i_sym])
    plt.xticks(x_pos + 0.5 * width,
               file_names,
               rotation = 'vertical',
               fontsize = 4)
    plt.legend((all_symbols), loc = 'upper center', ncol = 4)
    plt.savefig(data_set_name + ' freqs.png', dpi = 600)
    plt.close()
    
    
print_lengths(achu_file_contents,
              'Achu raw data',
              achu_file_names)
print_lengths(no_rep_achu_file_contents,
              'Achu no repetitions of symbol a',
              achu_file_names)
print_lengths(no_g_achu_file_contents,
              'Achu no g symbol',
              achu_file_names)
print_lengths(no_g_no_rep_achu_file_contents,
              'Achu no g symbol nor repetitions of symbol a',
              achu_file_names)
print_lengths(oldo_file_contents,
              'Oldo raw data',
              oldo_file_names)
print_lengths(no_rep_oldo_file_contents,
              'Oldo no repetitions of symbol a',
              oldo_file_names)
print_lengths(no_g_oldo_file_contents,
              'Oldo no g symbol',
              oldo_file_names)
print_lengths(no_g_no_rep_oldo_file_contents,
              'Oldo no g symbol nor repetitions of symbol a',
              oldo_file_names)

print_symbol_counts(achu_file_contents,
                    'Achu raw data',
                    achu_file_names)
print_symbol_counts(no_rep_achu_file_contents,
                    'Achu no repetitions of symbol a',
                    achu_file_names)
print_symbol_counts(no_g_achu_file_contents,
                    'Achu no g symbol',
                    achu_file_names)
print_symbol_counts(no_g_no_rep_achu_file_contents,
                    'Achu no g symbol nor repetitions of symbol a',
                    achu_file_names)
print_symbol_counts(oldo_file_contents,
                    'Oldo raw data',
                    oldo_file_names)
print_symbol_counts(no_rep_oldo_file_contents,
                    'Oldo no repetitions of symbol a',
                    oldo_file_names)
print_symbol_counts(no_g_oldo_file_contents,
                    'Oldo no g symbol',
                    oldo_file_names)
print_symbol_counts(no_g_no_rep_oldo_file_contents,
                    'Oldo no g symbol nor repetitions of symbol a',
                    oldo_file_names)


"""
print 'Raw achu data:'
for file_name in achu_file_names:
    print file_name + ' >' + achu_file_contents[file_name][:5] + '<'
print '\n'
    
print 'No repetition achu data:'        
for file_name in achu_file_names:
    print file_name + ' >' + no_rep_achu_file_contents[file_name][:5] + '<'
print '\n'

print 'No g achu data:'
for file_name in achu_file_names:
    print file_name + ' >' + no_g_achu_file_contents[file_name][:5] + '<' 
print '\n'

print 'No g nor rep achu data:'
for file_name in achu_file_names:
    print file_name + ' >' + no_g_no_rep_achu_file_contents[file_name][:5] + '<'
print '\n'

print 'Raw oldo data:'
for file_name in oldo_file_names:
    print file_name + ' >' + oldo_file_contents[file_name][:5] + '<'
print '\n'

print 'No rep oldo data:'
for file_name in oldo_file_names:
    print file_name + ' >' + no_rep_oldo_file_contents[file_name][:5] + '<'
print '\n'
"""