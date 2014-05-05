#ifndef FILE_READER_H
#define FILE_READER_H

#include<iostream>
#include<fstream>
#include<string>
#include<vector>

static void read_csv(const std::string & file_path,
                     std::vector<std::string> & content){
    std::ifstream input_file;
    input_file.open(file_path);
    if(input_file.is_open()){
        std::cout << "Reading from file " << file_path << std::endl;
    }else{
        std::cout << "Cannot read from file " << file_path << std::endl;
        return;
    }
    std::string temp_value;
    while(input_file.good())
    {
         getline(input_file, temp_value, ',');
         content.push_back(std::string(temp_value));
    }
    input_file.close();
}

static void read_csv(const std::string & file_path,
                     int* & content,
                     int & n_elts){
    std::vector<std::string> file_content;
    read_csv(file_path, file_content);
    n_elts = file_content.size();
    content = new int[n_elts];
    for(int i = 0; i < n_elts; ++i){
        content[i] = atoi(file_content[i].c_str());
    }
}

static void read_csv(const std::string & file_path,
                     char* & content,
                     int & n_elts){
    std::vector<std::string> file_content;
    read_csv(file_path, file_content);
    n_elts = file_content.size();
    content = new char[n_elts];
    for(int i = 0; i < n_elts; ++i){
        content[i] = file_content[i].c_str()[0];
    }
}

static void read_csv(const std::string & file_path,
                     std::string* & content,
                     int & n_elts){
    std::vector<std::string> file_content;
    read_csv(file_path, file_content);
    n_elts = file_content.size();
    content = new std::string[n_elts];
    for(int i = 0; i < n_elts; ++i){
        content[i] = file_content[i];
    }
}


#endif // FILE_READER_H
