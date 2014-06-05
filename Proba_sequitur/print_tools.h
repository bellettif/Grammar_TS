#ifndef PRINT_TOOLS_H
#define PRINT_TOOLS_H

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace print_tools{

template<typename T>
static void print_vect(const std::vector<T> & vect,
                       const std::string & prefix = ""){
    for(auto x : vect){
        std::cout << prefix << x << std::endl;
    }
}


template<typename T1, typename T2>
static void print_map(const std::unordered_map<T1, T2> & map,
                             const std::string & prefix = ""){
    std::vector<T1> key_set;
    for(auto xy : map){
        key_set.push_back(xy.first);
    }
    std::sort(key_set.begin(), key_set.end());
    for(auto x : key_set){
        std::cout << prefix << "key: " << x << " value: " << map.at(x) << std::endl;
    }
}

template<typename T1, typename T2>
static void print_map(const std::unordered_map<T1, std::vector<T2>> & map,
                      const std::string & prefix = ""){
    for(auto xy : map){
        std::cout << prefix << "key: " << xy.first << " values: ";
        for(auto z : xy.second){
            std::cout << z << " ";
        }std::cout << std::endl;
    }
}



}


#endif // PRINT_TOOLS_H
