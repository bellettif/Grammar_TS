'''
Created on 16 mai 2014

@author: francois
'''

import re
import load_data

sequence = ''.join(load_data.file_contents['achuSeq_1.csv'])

print sequence

def compute_counts(sequence, regex):
    print re.subn(regex, "", sequence)[1]
    return re.subn(regex, "", sequence)[1]

print compute_counts(sequence, "4.5")