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

typedef std::unordered_map<int_pair,
                            iter_pair_list,
                            pair_hash>
                                    central_access_map;


static const int MAX_TERMS = 100;
inline static size_t pair_hasher(const int_pair & x){
    return (size_t) (x.first * 100 + x.second);
}


class Mem_sandwich{

private:
    lateral_access_map              _first_maps;
    lateral_access_map              _second_maps;
    central_access_map              _center_lists;
    elt_list *                      _target_list;

public:
    Mem_sandwich():
        _first_maps(10, pair_hasher),
        _second_maps(10, pair_hasher),
        _center_lists(10, pair_hasher)
    {}

    void set_target_list(elt_list * target){
        _target_list = target;
    }

    void add_pair(const iter_pair & xy){
        int_pair xy_content = {xy.first->_content, xy.second->_content};
        if(_center_lists.count(xy_content) == 0){
            _center_lists.emplace(xy_content, iter_pair_list());
        }
        _center_lists.at(xy_content).push_back(xy);
        if(_first_maps.count(xy_content) == 0){
            _first_maps.emplace(xy_content, iter_iter_pair_iter_map(10, iter_hasher));
        }
        if(_second_maps.count(xy_content) == 0){
            _second_maps.emplace(xy_content, iter_iter_pair_iter_map(10, iter_hasher));
        }
        if(_second_maps.at(xy_content).count(xy.first) != 0){
            // Do not take overlapping pairs into account
            return;
        }
        _first_maps.at(xy_content)[xy.first] = std::prev(_center_lists.at(xy_content).end());
        _second_maps.at(xy_content)[xy.second] = std::prev(_center_lists.at(xy_content).end());
    }

    void remove_pair(const int_pair & xy,
                     const int & replacement){
        iter_pair_list & target_pairs = _center_lists[xy];
        iter_pair * current_pair;
        int_pair prev_content;
        iter_pair prev_iters;
        int_pair next_content;
        iter_pair next_iters;
        std::cout << "Removing pair" << std::endl;
        while(! target_pairs.empty()){
            std::cout << "Begin" << std::endl;
            current_pair = & target_pairs.front();
            std::cout << "Doing " << *(current_pair->first) << " " << *(current_pair->second) << std::endl;
            std::cout << "First" << std::endl;
            if(current_pair->first->_has_prev){
                // Deletion
                std::cout << "\tDeletion ";
                prev_content = {current_pair->first->_prev->_content,
                                current_pair->first->_content};
                delete_from_second(prev_content, current_pair->first);
                // Insertion
                std::cout << "\tInsertion ";
                prev_content = {current_pair->first->_prev->_content,
                                replacement};
                if(_center_lists.count(prev_content) == 0){
                    _center_lists.emplace(prev_content,
                                          iter_pair_list());
                    _first_maps.emplace(prev_content,
                                        iter_iter_pair_iter_map(10, iter_hasher));
                    _second_maps.emplace(prev_content,
                                         iter_iter_pair_iter_map(10, iter_hasher));
                }
                prev_iters = {current_pair->first->_prev,
                              current_pair->first};
                _center_lists.at(prev_content).push_back(prev_iters);
                _first_maps.at(prev_content)[prev_iters.first] = std::prev(_center_lists.at(prev_content).end());
                _second_maps.at(prev_content)[prev_iters.second] = std::prev(_center_lists.at(prev_content).end());
            }
            std::cout << "Second" << std::endl;
            if(current_pair->second->_has_next){
                // Deletion
                std::cout << "\tDeletion ";
                next_content = {current_pair->second->_content,
                                current_pair->second->_next->_content};
                delete_from_first(next_content, current_pair->second);
                // Insertion
                std::cout << "\tInsertion ";
                next_content = {replacement,
                                current_pair->second->_next->_content};
                if(_center_lists.count(next_content) == 0){
                    _center_lists.emplace(next_content,
                                          iter_pair_list());
                    _first_maps.emplace(next_content,
                                        iter_iter_pair_iter_map(10, iter_hasher));
                    _second_maps.emplace(next_content,
                                         iter_iter_pair_iter_map(10, iter_hasher));
                }
                next_iters = {current_pair->first,
                              current_pair->second->_next};
                _center_lists.at(next_content).push_back(prev_iters);
                _first_maps.at(next_content)[next_iters.first] = std::prev(_center_lists.at(next_content).end());
                _second_maps.at(next_content)[next_iters.second] = std::prev(_center_lists.at(next_content).end());
            }
            std::cout << "Erasing pair" << std::endl;
            current_pair->first->_content = replacement;
            _target_list->erase(current_pair->second);
            target_pairs.pop_front();
            std::cout << "End" << std::endl;
            std::cout << std::endl;
        }
        std::cout << "Deletion done" << std::endl;
     }

    void delete_from_first(const int_pair & content,
                           const iter & target){
        std::cout <<"\t\t\t" << _first_maps.count(content) << std::endl;
        if(_first_maps.at(content).count(target) == 0){
            return; // Nothing to do, only non overlapping pairs are taken into account
        }
        iter_pair_iter & to_delete = _first_maps.at(content).at(target);
        _center_lists.at(content).erase(to_delete);
        _second_maps.at(content).erase(to_delete->second);
        _first_maps.at(content).erase(target);
    }

    void delete_from_second(const int_pair & content,
                            const iter & target){
        std::cout <<"\t\t\t" << _second_maps.count(content) << std::endl;
        if(_second_maps.at(content).count(target) == 0){
            return; // Nothing to do, only non overlapping pairs are taken into account
        }
        iter_pair_iter & to_delete = _second_maps.at(content).at(target);
        _center_lists.at(content).erase(to_delete);
        _first_maps.at(content).erase(to_delete->first);
        _second_maps.at(content).erase(target);
    }

    void print(const int_pair & target_pair){
        if(_center_lists.count(target_pair) == 0){
            std::cout << "Pair is not present" << std::endl;
        }
        std::cout << "Printing center list" << std::endl;
        for(auto x : _center_lists.at(target_pair)){
            std::cout << "{" << *(x.first) << ", " << *(x.second) << "} ";
        }std::cout << std::endl;
        std::cout << "Printing first map for pair "
                  << target_pair.first << " " << target_pair.second << std::endl;
        for(auto xy : _first_maps.at(target_pair)){
            std::cout << "First: " << *(xy.first) << ", pair: "
                      << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
        }std::cout << std::endl;
        std::cout << "Pringint second map for pair "
                  << target_pair.first << " " << target_pair.second << std::endl;
        for(auto xy : _second_maps.at(target_pair)){
            std::cout << "Second: " << *(xy.first) << ", pair: "
                      << "{" << *(xy.second->first) << ", " << *(xy.second->second) << "} ";
        }std::cout << std::endl;
    }

    const lateral_access_map & get_first_maps() const{
        return _first_maps;
    }

    const lateral_access_map & get_second_maps() const{
        return _second_maps;
    }

    const central_access_map & get_central_lists() const{
        return _center_lists;
    }

};


#endif // MEM_SANDWICH_H
