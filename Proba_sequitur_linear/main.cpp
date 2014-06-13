#include <unordered_map>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>

#include "file_reader.h"
#include "proba_sequitur.h"
#include "mem_sandwich.h"
#include "launcher.h"

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

static const std::vector<std::string> ACHU_FILE_NAMES = {"achuSeq_1.csv",
                                                         "achuSeq_2.csv",
                                                         "achuSeq_3.csv",
                                                         "achuSeq_4.csv",
                                                         "achuSeq_5.csv",
                                                         "achuSeq_6.csv",
                                                         "achuSeq_7.csv",
                                                         "achuSeq_8.csv",
                                                         "achuSeq_9.csv"};

static const std::vector<std::string> OLDO_FILE_NAMES = {"oldoSeq_1.csv",
                                                         "oldoSeq_2.csv",
                                                         "oldoSeq_3.csv",
                                                         "oldoSeq_4.csv",
                                                         "oldoSeq_5.csv",
                                                         "oldoSeq_6.csv",
                                                         "oldoSeq_8.csv",
                                                         "oldoSeq_9.csv",
                                                         "oldoSeq_10.csv"};


int main(){
    string_vect_vect achu_oldo_content;
    for(const std::string & file_name : FILE_NAMES){
        achu_oldo_content.push_back(file_reader::read_csv(FOLDER_PATH + file_name).front());
    }

    string_vect_vect achu_content;
    for(const std::string & file_name : ACHU_FILE_NAMES){
        achu_content.push_back(file_reader::read_csv(FOLDER_PATH + file_name).front());
    }

    string_vect_vect oldo_content;
    for(const std::string & file_name : OLDO_FILE_NAMES){
        oldo_content.push_back(file_reader::read_csv(FOLDER_PATH + file_name).front());
    }

    string_int_map to_index_map;
    int_string_map to_string_map;

    int_vect_vect achu_oldo_translation_result;
    file_reader::translate_to_ints(achu_oldo_content,
                                   to_index_map,
                                   to_string_map,
                                   achu_oldo_translation_result);


    int_vect_vect achu_translation_result;
    file_reader::translate_to_ints(achu_content,
                                   to_index_map,
                                   to_string_map,
                                   achu_translation_result);

    int_vect_vect oldo_translation_result;
    file_reader::translate_to_ints(oldo_content,
                                   to_index_map,
                                   to_string_map,
                                   oldo_translation_result);

    Proba_sequitur ps (6,
                       40,
                       achu_oldo_translation_result,
                       achu_oldo_translation_result,
                       to_index_map,
                       to_string_map);

    time_t t = clock();
    ps.run();
    t = clock() - t;
    std::cout << ((double) t) / ((double) CLOCKS_PER_SEC) << std::endl;

    ps.print_counts();

    string_vect_vect result = ps.translate_inference_samples();

    return 0;
}
