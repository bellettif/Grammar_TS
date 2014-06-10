#ifndef MEM_SANDWICH_H
#define MEM_SANDWICH_H

#include <unordered_map>
#include <list>
#include <string>

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

static const int MAX_LENGTH = 10000;
inline static size_t iter_hasher (const iter & x){
    return (size_t) x->_seq_index * MAX_LENGTH + x->_word_index;
}

typedef std::pair<int, int>         int_pair;
typedef std::function<size_t(const int_pair &)> pair_hash;
typedef std::unordered_map<int_pair,
                           iter_iter_pair_iter_map,
                           pair_hash>
                                    lateral_access_map;

static const int MAX_TERMS = 100;
inline static size_t pair_hasher(const int_pair & x){
    return (size_t) (x.first * 100 + x.second);
}


class Mem_sandwich{

private:
    lateral_access_map              _first_maps;
    lateral_access_map              _second_maps;
    iter_pair_list                  _center_list;
    elt_list *                      _target_list;

public:
    Mem_sandwich():
        _first_maps(10, pair_hasher),
        _second_maps(10, pair_hasher)
    {}

    void set_target_list(elt_list * target){
        _target_list = target;
    }

    void add_pair(const iter_pair & xy){
        int_pair xy_content = {xy.first->_content, xy.second->_content};
        _center_list.push_back(xy);
        if(_first_maps.count(xy_content) == 0){
            _first_maps.emplace(xy_content, iter_iter_pair_iter_map(MAX_LENGTH, iter_hasher));
        }
        if(_second_maps.count(xy_content) == 0){
            _second_maps.emplace(xy_content, iter_iter_pair_iter_map(MAX_LENGTH, iter_hasher));
        }
        if(_second_maps.at(xy_content).count(xy.first) != 0){
            // Do not take overlapping pairs into account
            return;
        }
        _first_maps.at(xy_content)[xy.first] = std::prev(_center_list.end());

        _second_maps.at(xy_content)[xy.second] = std::prev(_center_list.end());
    }

    void print(const int_pair & target_pair){
        std::cout << "Printing center list" << std::endl;
        for(auto x : _center_list){
            std::cout << "{" << *(x.first) << ", " << *(x.second) << "} ";
        }std::cout << std::endl;
        std::cout << "Printing first map for pair "
                  << target_pair.first << " " << target_pair.second << std::endl;
        if(_first_maps.count(target_pair) != 0){
            for(auto xy : _first_maps.at(target_pair)){
                std::cout << "First: " << *(xy.first) << ", pair: "
                          << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
            }std::cout << std::endl;
        }else{
            std::cout << "Pair is not present" << std::endl;
        }
        std::cout << "Pringint second map for pair "
                  << target_pair.first << " " << target_pair.second << std::endl;
        if(_second_maps.count(target_pair) != 0){
            for(auto xy : _second_maps.at(target_pair)){
                std::cout << "Second: " << *(xy.first) << ", pair: "
                          << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
            }std::cout << std::endl;
        }else{
            std::cout << "Pair is not present" << std::endl;
        }
    }

    const lateral_access_map & get_first_maps() const{
        return _first_maps;
    }

    const lateral_access_map & get_second_maps() const{
        return _second_maps;
    }

    const iter_pair_list & get_central_list() const{
        return _center_list;
    }

};


#endif // MEM_SANDWICH_H
