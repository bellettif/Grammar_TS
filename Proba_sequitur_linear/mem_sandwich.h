#ifndef MEM_SANDWICH_H
#define MEM_SANDWICH_H

#include <unordered_map>
#include <list>

#include "element.h"

typedef Element                     elt;
typedef std::list<Element>          elt_list;
typedef elt_list::iterator          iter;

typedef std::pair<iter, iter>       iter_pair;
typedef std::list<iter_pair>        iter_pair_list;
typedef iter_pair_list::iterator    iter_pair_iter;

typedef std::function<size_t(const iter &)> iter_hash;

typedef std::unordered_map<iter,
                           iter_pair_iter,
                           iter_hash>
                                    iter_iter_pair_iter_map;

static const int MAX_LENGTH = 100;
inline static size_t hasher (const iter & x){
    return (size_t) x->_seq_index * MAX_LENGTH + x->_word_index;
}

class Mem_sandwich{

private:
    iter_iter_pair_iter_map         _first_map;
    iter_iter_pair_iter_map         _second_map;
    iter_pair_list                  _center_list;

public:
    Mem_sandwich():
        _first_map(MAX_LENGTH, hasher),
        _second_map(MAX_LENGTH, hasher)
    {}

    void add_pair(const iter_pair & x){
        _center_list.push_back(x);
        _first_map[x.first] = std::prev(_center_list.end());
        _second_map[x.second] = std::prev(_center_list.end());
    }

    void print(){
        std::cout << "Printing center list" << std::endl;
        for(auto x : _center_list){
            std::cout << "{" << *(x.first) << ", " << *(x.second) << "} ";
        }std::cout << std::endl;
        std::cout << "Printing first map" << std::endl;
        for(auto xy : _first_map){
            std::cout << "First: " << *(xy.first) << ", pair: "
                      << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
        }std::cout << std::endl;
        std::cout << "Pringint second map" << std::endl;
        for(auto xy : _second_map){
            std::cout << "Second: " << *(xy.first) << ", pair: "
                      << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
        }std::cout << std::endl;
    }

};


#endif // MEM_SANDWICH_H
