'''
Created on 26 mai 2014

@author: francois
'''

from Proba_sequitur import Proba_sequitur

import load_data

print 'Achu file contents with repetitions'
proba_seq = Proba_sequitur(load_data.achu_file_contents.values(), True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Achu filtered file contents with repetitions'
proba_seq = Proba_sequitur(load_data.filtered_achu_file_contents.values(), True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Achu file contents without repetitions'
proba_seq = Proba_sequitur(load_data.achu_file_contents.values(), False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Achu filtered file contents without repetitions'
proba_seq = Proba_sequitur(load_data.filtered_achu_file_contents.values(), False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Oldo filtered file contents without repetitions'
proba_seq = Proba_sequitur(load_data.oldo_file_contents.values(), True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Oldo filtered file contents with repetitions'
proba_seq = Proba_sequitur(load_data.filtered_oldo_file_contents.values(), True)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Oldo file contents without repetitions'
proba_seq = Proba_sequitur(load_data.oldo_file_contents.values(), False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'

print 'Oldo filtered file contents without repetitions'
proba_seq = Proba_sequitur(load_data.filtered_oldo_file_contents.values(), False)
proba_seq.infer_grammar(6)
proba_seq.print_result()
print '\n'