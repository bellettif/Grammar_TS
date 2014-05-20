'''
Created on 19 mai 2014

@author: francois
'''

import load_data
import re

file_contents = load_data.file_contents

for file_name, file_content in file_contents.iteritems():
    if 'achu' not in file_name: continue
    print 'File %s' % file_name
    filtered_sequence = re.subn("1*", "1", ''.join(file_content))[0]
    print 'Number of symbols: %d' % len(filtered_sequence) 
    print re.subn("1*", "1", ''.join(file_content))[0]
    print '\n'