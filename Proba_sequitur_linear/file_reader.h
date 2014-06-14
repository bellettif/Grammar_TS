#ifndef FILE_READER_H
#define FILE_READER_H

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<unordered_map>

#include "string_utils.h"
#include "decision_making.h"

typedef std::vector<char> char_vector;
typedef std::vector<int>  int_vect;

namespace file_reader{

static const std::unordered_map<char, char> translation_map({ {'1', 'a'},
                                                              {'2', 'b'},
                                                              {'3', 'c'},
                                                              {'4', 'd'},
                                                              {'5', 'e'},
                                                              {'6', 'f'},
                                                              {'7', 'g'}});

static inline void translate(char_vector & input){
    for(int i = 0; i < input.size(); ++i){
        input[i] = translation_map.at(input[i]);
    }
}

static std::vector<std::string> read_lines_from_file(const std::string & file_path){
    std::ifstream input_file;
    input_file.open(file_path);
    if(input_file.is_open()){
        std::cout << "Reading from file " << file_path << std::endl;
    }else{
        std::cout << "Cannot read from file " << file_path << std::endl;
        return std::vector<std::string>();
    }
    std::string temp_value;
    std::vector<std::string> content;
    while(input_file.good())
    {
         getline(input_file, temp_value);
         if (temp_value == "") continue;
         content.push_back(std::string(temp_value));
    }
    input_file.close();
    return content;
}

static std::vector<std::vector<std::string>> read_csv(const std::string & file_path){
    std::vector<std::string> file_content = read_lines_from_file(file_path);
    std::vector<std::vector<std::string>> result (file_content.size());
    for(int i = 0; i < file_content.size(); ++i){
        string_utils::split(file_content[i],
                            ",",
                            result[i]);
    }
    return result;
}

static void translate_to_ints(const std::vector<std::string> & input,
                              const std::unordered_map<std::string, int> & to_index_map,
                              std::vector<int> & translation_result){
    translation_result = std::vector<int>(input.size());
    for(int i = 0; i < input.size(); ++i){
        translation_result[i] = to_index_map.at(input[i]);
    }
}

static void translate_to_ints(const std::vector<std::vector<std::string>> & input,
                              std::unordered_map<std::string, int> & to_index_map,
                              std::unordered_map<int, std::string> & to_string_map,
                              std::vector<std::vector<int>> & translation_result){
    int n_words = to_index_map.size() + 1;
    for(const std::vector<std::string> & line : input){
        for(const std::string & word : line){
            if(to_index_map.count(word) == 0){
                to_index_map[word] = n_words;
                to_string_map[n_words] = word;
                ++ n_words;
            }
        }
    }
    translation_result = std::vector<std::vector<int>> (input.size());
    for(int i = 0; i < input.size(); ++i){
        translate_to_ints(input[i],
                          to_index_map,
                          translation_result[i]);
    }
}

static int_vect sub_selection(const int_vect & input,
                              const double & proba,
                              RNG & rng = core_rng){
    std::bernoulli_distribution distrib(1.0 - proba);
    int_vect result;
    for(auto x : input){
        if(distrib(rng)){
            result.push_back(x);
        }
    }
    return result;
}

}

#endif // FILE_READER_H
