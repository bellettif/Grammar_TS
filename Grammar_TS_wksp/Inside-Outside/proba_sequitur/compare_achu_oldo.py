'''
Created on 29 mai 2014

@author: francois
'''

import cPickle as pickle

builds = ['achu_and_oldo',
          'achu',
          'oldo']

count_targets = ['achu',
          'oldo']

all_extracts = []
for build in builds:
    for count in count_targets:
        file_path = 'Achu_oldo_grammars/' + build + '_count_' + count + '_rep_6.pi'
        all_extracts.append(pickle.load(open(file_path, 'rb')))

all_hash_codes = []
for x in all_extracts:
    all_hash_codes.extend(x['hashcode_to_rule'].keys())

counts = {}
for extract in all_extracts:
    hashcode_to_rule = x['hashcode_to_rule']
    rule_to_hashcode = x['rule_to_hashcode']
    temp_counts = x['counts']
    for rule, count_dict in temp_counts.iteritems():
        hashcode = rule_to_hashcode[rule]
        if hashcode not in counts:
            counts[hashcode] = {}
        for i, count in count_dict.iteritems():
            if i not in counts[hashcode]:
                counts[hashcode][i] = 0
            counts[hashcode][i] += count
print counts

    