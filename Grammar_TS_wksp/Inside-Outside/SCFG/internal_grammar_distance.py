'''
Created on 29 mai 2014

@author: francois
'''

import numpy as np
from matplotlib import pyplot as plt
import copy
import multiprocessing as multi
import time


def compute_KL_signature(first_sign, second_sign):
    if set(first_sign.keys()) != set(second_sign.keys()):
        return np.inf
    else:
        result = 0
        for key in first_sign.keys():
            p = first_sign[key]
            q = second_sign[key]
            result += p * np.log(p / q) + q * np.log(q / p)
        return result

N_PROCESSES = 6

def compute_distance(left_grammar,
                     right_grammar,
                     n_samples,
                     max_length = 0):
    if sorted(left_grammar.term_chars) != sorted(right_grammar.term_chars):
        return np.inf
    left_samples = left_grammar.produce_sentences(n_samples, max_length)
    right_samples = right_grammar.produce_sentences(n_samples, max_length)
    left_right_probas = left_grammar.estimate_likelihoods(right_samples)
    left_left_probas = left_grammar.estimate_likelihoods(left_samples)
    right_left_probas = right_grammar.estimate_likelihoods(left_samples)
    right_right_probas = right_grammar.estimate_likelihoods(right_samples)
    #
    selection = np.where(left_left_probas != 0)
    left_result = np.sum(np.log(left_left_probas[selection] / right_left_probas[selection]) * left_left_probas[selection])
    left_result /= float(n_samples)
    #
    selection = np.where(right_right_probas != 0)
    right_result = np.sum(np.log(right_right_probas[selection] / left_right_probas[selection]) * right_right_probas[selection])
    right_result /= float(n_samples)
    return left_result + right_result

def compute_distance_tuple(input_tuple):
    return compute_distance(input_tuple[0],
                            input_tuple[1],
                            input_tuple[2],
                            input_tuple[3])

def compute_distance_MC(left_grammar,
                        right_grammar,
                        n_samples,
                        max_length = 0,
                        epsilon = 0):
    if sorted(left_grammar.term_chars) != sorted(right_grammar.term_chars):
        return np.inf
    left_signature = left_grammar.compute_signature(n_samples,
                                                     max_length = max_length)
    right_signature = right_grammar.compute_signature(n_samples,
                                                       max_length = max_length)
    merged_signature = left_signature.items() + right_signature.items()
    if max_length == 0 and epsilon != 0:
        merged_signature.sort(key = (lambda x : -x[1]))
        total_counts = float(sum([x[1] for x in merged_signature]))
        current_total = 0
        max_length = 0
        for sentence, count in merged_signature:
            current_total += count / total_counts
            if current_total >= 1.0 - epsilon:
                max_length = len(sentence.split(' '))
                break
    left_signature = dict(filter(lambda x : len(x[0].split(' ')) <= max_length, 
                                 left_signature.items()))
    right_signature = dict(filter(lambda x : len(x[0].split(' ')) <= max_length,
                                  right_signature.items()))
    return compute_KL_signature(left_signature,
                                right_signature)

def compute_distance_MC_tuple(input_tuple):
    return compute_distance_MC(input_tuple[0],
                               input_tuple[1],
                               input_tuple[2],
                               input_tuple[3],
                               input_tuple[4])

def compute_distance_matrix(left_grammar, right_grammar, n_samples):
    n_rules_left = left_grammar.N
    n_rules_right = right_grammar.N
    instruction_set = []
    for i in xrange(n_rules_left):
        for j in xrange(n_rules_right):
            left_copy = copy.deepcopy(left_grammar)
            right_copy = copy.deepcopy(right_grammar)
            left_copy.rotate(i)
            right_copy.rotate(j)
            instruction_set.append((left_copy, right_copy, n_samples, 0))
    p = multi.Pool(processes = N_PROCESSES)
    distances = p.map(compute_distance_tuple, instruction_set)
    distances = np.asarray(distances, dtype = np.double)
    distances = np.reshape(distances, (n_rules_left, n_rules_right))
    return distances

def compute_internal_distance_matrix_MC(left_grammar, right_grammar, n_samples, epsilon):
    n_rules_left = left_grammar.N
    n_rules_right = right_grammar.N
    instruction_set = []
    for i in xrange(n_rules_left):
        for j in xrange(n_rules_right):
            left_copy = copy.deepcopy(left_grammar)
            right_copy = copy.deepcopy(right_grammar)
            left_copy.rotate(i)
            right_copy.rotate(j)
            instruction_set.append((left_copy, right_copy, n_samples, 0, epsilon))
    p = multi.Pool(processes = N_PROCESSES)
    distances = p.map(compute_distance_MC_tuple, instruction_set)
    distances = np.asarray(distances, dtype = np.double)
    distances = np.reshape(distances, (n_rules_left, n_rules_right))
    return distances        