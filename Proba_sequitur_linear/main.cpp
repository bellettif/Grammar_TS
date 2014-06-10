#include <unordered_map>
#include <iostream>
#include <string>
#include <vector>

#include "file_reader.h"
#include "proba_sequitur.h"
#include "mem_sandwich.h"

static const std::string FOLDER_PATH = "/Users/francois/Grammar_TS/data/";
static const std::vector<std::string> FILE_NAMES = {"achuSeq_1.csv",
                                                    "achuSeq_2.csv",
                                                    "achuSeq_3.csv",
                                                    "achuSeq_4.csv",
                                                    "achuSeq_5.csv",
                                                    "achuSeq_6.csv",
                                                    "achuSeq_7.csv",
                                                    "achuSeq_8.csv",
                                                    "achuSeq_9.csv",
                                                    "oldoSeq_1.csv",
                                                    "oldoSeq_2.csv",
                                                    "oldoSeq_3.csv",
                                                    "oldoSeq_4.csv",
                                                    "oldoSeq_5.csv",
                                                    "oldoSeq_6.csv",
                                                    "oldoSeq_8.csv",
                                                    "oldoSeq_9.csv",
                                                    "oldoSeq_10.csv"};

int main(){
    std::vector<std::vector<std::string>> content;
    for(const std::string & file_name : FILE_NAMES){
        content.push_back(file_reader::read_csv(FOLDER_PATH + file_name).front());
    }

    std::unordered_map<std::string, int> to_index_map;
    std::unordered_map<int, std::string> to_string_map;
    std::vector<std::vector<int>> translation_result;

    file_reader::translate_to_ints(content,
                                   to_index_map,
                                   to_string_map,
                                   translation_result);

    Proba_sequitur ps (translation_result,
                       translation_result,
                       to_index_map,
                       to_string_map);

    ps.print_bare_lks();

    ps.compute_pattern_scores();

    ps.print_pattern_scores();

    return 0;
}
