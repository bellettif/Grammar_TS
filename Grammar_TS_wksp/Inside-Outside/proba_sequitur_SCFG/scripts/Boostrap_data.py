'''
Created on 4 juin 2014

@author: francois
'''

import cPickle as pickle

import load_data

achu_data = load_data.achu_file_contents
oldo_data = load_data.oldo_file_contents

f_achu_data = load_data.filtered_achu_file_contents
f_oldo_data = load_data.filtered_oldo_file_contents

pickle.dump(achu_data, open('../data/achu_data.pi', 'wb'))
pickle.dump(oldo_data, open('../data/oldo_data.pi', 'wb'))

pickle.dump(f_achu_data, open('../data/f_achu_data.pi', 'wb'))
pickle.dump(f_oldo_data, open('../data/f_oldo_data.pi', 'wb'))
