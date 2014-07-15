'''
Created on 15 juil. 2014

@author: francois
'''

from SCFG.sto_grammar import SCFG
import re

import numpy as np
from matplotlib import pyplot as plt

left_reg_ex_rules = {0 : ([[1, 2], [9, 4], [9, 6]], #root
              [0.9, 0.05, 0.05],
              [],
              []),
         1 : ([], # A
              [],
              ['a', 'b', 'c', 'd'],
              [0.85, 0.05, 0.05, 0.05]),
         2 : ([[9, 2], [3, 4], [9, 6]], # rule_1
              [0.05, 0.9, 0.05],
              [],
              []),
         3 : ([], # B
              [],
              ['a', 'b', 'c', 'd'],
              [0.05, 0.85, 0.05, 0.05]),
         4 : ([[9, 2], [9, 4], [5, 6]], # rule_2
              [0.05, 0.05, 0.9],
              [],
              []),
         5 : ([], # C
              [],
              ['a', 'b', 'c', 'd'],
              [0.05, 0.05, 0.85, 0.05]),
         6 : ([[1, 2], [9, 4], [9, 6], [7, 8]], # rule_3
              [0.8, 0.05, 0.05, 0.1],
              [],
              []),
         7 : ([], # D
              [],
              ['a', 'b', 'c', 'd'],
              [0.05, 0.05, 0.05, 0.85]),
         8 : ([], # rule_4
              [],
              ['end'],
              [1.0]),
         9 : ([], # W (noise)
              [],
              ['a', 'b', 'c', 'd'],
              [0.25, 0.25, 0.25, 0.25])
        }
left_reg_ex = SCFG()
left_reg_ex.init_from_rule_dict(left_reg_ex_rules)

left_reg_ex.draw_grammar('Left_reg_ex.png')
left_reg_ex.draw_grammar('Left_reg_ex_th.png', threshold=0.08)

example_sentences = left_reg_ex.produce_sentences(10000)

term_chars = left_reg_ex.term_chars
n_term_chars = len(term_chars)

all_pairs = [x + y for x in term_chars for y in term_chars]

print all_pairs

sentences = [''.join(x) for x in example_sentences]

co_occurrence_dict = {}
for pair in all_pairs:
    if pair not in co_occurrence_dict:
        co_occurrence_dict[pair] = 0
    for sentence in sentences:
        co_occurrence_dict[pair] += len(re.findall(pair, sentence))

total = float(sum(co_occurrence_dict.values()))

co_oc_matrix = np.zeros((n_term_chars, n_term_chars))
for i, left in enumerate(term_chars):
    for j, right in enumerate(term_chars):
        co_oc_matrix[i, j] = co_occurrence_dict[left + right] / total
        
plt.matshow(co_oc_matrix, cmap = 'gray')
plt.yticks(range(n_term_chars), term_chars)
plt.xticks(range(n_term_chars), term_chars)
plt.title('Co occurrences of pairs of symbols')
plt.ylabel('Predecessor')
plt.xlabel('Successor')
plt.colorbar()
plt.savefig('Co-occurrences left reg examples.png', dpi = 600)



