#ifndef FILE_READER_H
#define FILE_READER_H

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<unordered_map>

namespace file_reader{

static const std::unordered_map<std::string, std::string> translation_map({ {"1", "a"},
                                                                            {"2", "b"},
                                                                            {"3", "c"},
                                                                            {"4", "d"},
                                                                            {"5", "e"},
                                                                            {"6", "f"},
                                                                            {"7", "g"}
                                                                          });

static inline std::string translate_to_chars(std::string input){
    std::string output (input);
    for(auto xy : translation_map){
        string_utils::replace(output,
                              xy.first,
                              xy.second);
    }
    return output;
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
         content.push_back(std::string(temp_value));
    }
    input_file.close();
    return content;
}

static std::vector<std::string> read_csv(const std::string & file_path){
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
         getline(input_file, temp_value, ',');
         content.push_back(std::string(temp_value));
    }
    input_file.close();
    return content;
}

static void read_csv(const std::string & file_path,
                     std::vector<std::string> & content){
    content = read_csv(file_path);
}

static void read_csv(const std::string & file_path,
                     std::vector<int> & content){
    std::vector<std::string> file_content = read_csv(file_path);
    int n_elts = file_content.size();
    content = std::vector<int>(n_elts);
    for(int i = 0; i < n_elts; ++i){
        content[i] = atoi(file_content[i].c_str());
    }
}

static void read_csv(const std::string & file_path,
                     std::vector<char> content){
    std::vector<std::string> file_content = read_csv(file_path);
    int n_elts = file_content.size();
    content = std::vector<char>(n_elts);
    for(int i = 0; i < n_elts; ++i){
        content[i] = file_content[i].c_str()[0];
    }
}

template<typename T>
static std::vector<T> eliminate_repetitions(std::vector<T> content){
    std::vector<T> buffer;
    T left = content[0];
    buffer.push_back(left);
    T right;
    for(int i = 1; i < content.size(); ++i){
        right = content[i];
        if(right != left){
            buffer.push_back(right);
        }
        left = right;
    }
    return buffer;
}

template<typename T>
static std::vector<T> eliminate_elt(std::vector<T> content,
                                    const T & target){
    std::vector<T> buffer;
    for(int i = 0; i < content.size(); ++i){
        if(content[i] != target){
            buffer.push_back(content[i]);
        }
    }
    return buffer;
}

}

#endif // FILE_READER_H
