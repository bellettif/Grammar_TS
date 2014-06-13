#ifndef LAUNCHER_H
#define LAUNCHER_H

#include "proba_sequitur.h"
#include "file_reader.h"

void launch_proba_sequitur(const string_vect_vect & inference_content,
                           const string_vect_vect & count_content,
                           int degree, int max_rules,
                           string_vect_vect & inference_parsed,
                           string_vect_vect & counts_parsed,
                           string_vect & hashcodes,
                           string_pair_vect & hashed_rhs,
                           double_vect_vect & relative_counts,
                           int_vect_vect & absolute_counts,
                           int_vect & levels,
                           int_vect & depths){

    string_int_map to_index_map;
    int_string_map to_string_map;

    int_vect_vect translated_inference_content;
    int_vect_vect translated_count_content;
    file_reader::translate_to_ints(inference_content,
                                   to_index_map,
                                   to_string_map,
                                   translated_inference_content);
    file_reader::translate_to_ints(count_content,
                                   to_index_map,
                                   to_string_map,
                                   translated_count_content);

    Proba_sequitur ps (degree,
                       max_rules,
                       translated_inference_content,
                       translated_count_content,
                       to_index_map,
                       to_string_map);
    ps.run();

    inference_parsed = ps.translate_inference_samples();
    counts_parsed = ps.translate_counting_samples();

    ps.to_hashed_vectors(hashcodes,
                      hashed_rhs,
                      relative_counts,
                      absolute_counts,
                      levels,
                      depths);
}



#endif // LAUNCHER_H
