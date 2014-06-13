#ifndef MEM_SANDWICH_H
#define MEM_SANDWICH_H

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <string>

#include "element.h"

typedef Element                                     elt;
typedef std::list<Element>                          elt_list;
typedef elt_list::iterator                          iter;

typedef std::unordered_map<int, std::string>        int_string_map;
typedef std::unordered_map<std::string, int>        string_int_map;

typedef std::pair<iter, iter>                       iter_pair;
typedef std::list<iter_pair>                        iter_pair_list;
typedef iter_pair_list::iterator                    iter_pair_iter;

typedef std::function<size_t(const iter &)>         iter_hash;
typedef std::function<size_t(const iter_pair &)>    iter_pair_hash;

typedef std::unordered_map<iter,
                           iter_pair_iter,
                           iter_hash>
                                    iter_iter_pair_iter_map;
typedef std::unordered_map<iter,
                           iter_pair,
                           iter_hash>
                                    iter_iter_pair_map;

typedef std::unordered_set<iter_pair,
                           iter_pair_hash>
                                    iter_pair_set;

static const int MAX_LENGTH = 10000;
inline static size_t iter_hasher (const iter & x){
    return (size_t) x->_word_index;
}

inline static size_t iter_pair_hasher(const iter_pair & xy){
    return (size_t) xy.first->_word_index * MAX_LENGTH +
                    xy.second->_word_index;
}

typedef std::pair<int, int>                     int_pair;
typedef std::function<size_t(const int_pair &)> pair_hash;
typedef std::unordered_map<int_pair,
                           iter_iter_pair_iter_map,
                           pair_hash>
                                    lateral_access_map;

typedef std::unordered_map<int_pair,
                            iter_pair_list,
                            pair_hash>
                                    central_access_map;

typedef std::unordered_map<int_pair,
                           iter_pair_set,
                           pair_hash>
                                    overlap_access_map;

static const int MAX_TERMS = 100;
inline static size_t pair_hasher(const int_pair & x){
    return (size_t) (x.first * 100 + x.second);
}


class Mem_sandwich{

private:
    lateral_access_map              _first_maps;
    lateral_access_map              _second_maps;
    central_access_map              _center_lists;
    overlap_access_map              _seen;
    overlap_access_map              _masked;
    iter_pair_list                  _not_overlapping_anymore;
    elt_list *                      _target_list;

public:
    Mem_sandwich():
        _first_maps(10, pair_hasher),
        _second_maps(10, pair_hasher),
        _center_lists(10, pair_hasher),
        _seen(10, pair_hasher),
        _masked(10, pair_hasher)
    {}

    void set_target_list(elt_list * target){
        _target_list = target;
    }

    void add_pair(const iter_pair & xy){
        int_pair xy_content = {xy.first->_content, xy.second->_content};
        if(_center_lists.count(xy_content) == 0){
            _center_lists.emplace(xy_content, iter_pair_list());
            _first_maps.emplace(xy_content, iter_iter_pair_iter_map(10, iter_hasher));
            _second_maps.emplace(xy_content, iter_iter_pair_iter_map(10, iter_hasher));
            _seen.emplace(xy_content, iter_pair_set(10, iter_pair_hasher));
            _masked.emplace(xy_content, iter_pair_set(10, iter_pair_hasher));
        }
        if(_second_maps.at(xy_content).count(xy.first) == 0){
            _seen.at(xy_content).insert(xy);
        }else{
            const iter_pair & previous_pair = *(_second_maps.at(xy_content).at(xy.first));
            if(_masked.at(xy_content).count(previous_pair) == 0){
                _masked.at(xy_content).insert(xy);
            }else{
                _seen.at(xy_content).insert(xy);
            }
        }
        _center_lists.at(xy_content).push_back(xy);
        _first_maps.at(xy_content)[xy.first] = std::prev(_center_lists.at(xy_content).end());
        _second_maps.at(xy_content)[xy.second] = std::prev(_center_lists.at(xy_content).end());
    }

    void remove_pair(const int_pair & xy,
                     const int & replacement){
        iter_pair_list & target_pairs =_center_lists[xy];
        iter_pair * current_pair;
        int_pair prev_content;
        iter_pair prev_iters;
        int_pair next_content;
        iter_pair next_iters;
        _not_overlapping_anymore.clear();
        while(!target_pairs.empty() ){
            current_pair = new iter_pair(target_pairs.front());
            target_pairs.pop_front();
            //if(_masked.at(xy).count(*current_pair) > 0) continue;
            //std::cout << "Deleting pair" << *(current_pair->first) << " " << *(current_pair->second) << std::endl;
            if(current_pair->first != _target_list->begin()){
                // Deletion
                prev_content = {std::prev(current_pair->first)->_content,
                                current_pair->first->_content};
                prev_iters = {std::prev(current_pair->first),
                              current_pair->first};
                //std::cout << "\tDeleting from second " << *(std::prev(current_pair->first)) << " " << *(current_pair->first) << std::endl;
                delete_from_second(prev_content, prev_iters);
                // Insertion
                prev_content = {std::prev(current_pair->first)->_content,
                                replacement};
                if(_center_lists.count(prev_content) == 0){
                    _center_lists.emplace(prev_content,
                                          iter_pair_list());
                    _first_maps.emplace(prev_content,
                                        iter_iter_pair_iter_map(10, iter_hasher));
                    _second_maps.emplace(prev_content,
                                         iter_iter_pair_iter_map(10, iter_hasher));
                    _seen.emplace(prev_content,
                                            iter_pair_set(10, iter_pair_hasher));
                    _masked.emplace(prev_content,
                                            iter_pair_set(10, iter_pair_hasher));
                }
                _center_lists.at(prev_content).push_back(prev_iters);
                iter_pair_iter temp_prev = std::prev(_center_lists.at(prev_content).end());
                _first_maps.at(prev_content)[prev_iters.first] = temp_prev;
                _second_maps.at(prev_content)[prev_iters.second] = temp_prev;
                if(_second_maps.at(prev_content).count(prev_iters.first) == 0){
                    _seen.at(prev_content).insert(prev_iters);
                }else{
                    const iter_pair & previous_pair = *(_second_maps.at(prev_content).at(prev_iters.first));
                    if(_masked.at(prev_content).count(previous_pair) == 0){
                        _masked.at(prev_content).insert(prev_iters);
                    }else{
                        _seen.at(prev_content).insert(prev_iters);
                    }
                }
            }
            if(current_pair->second != std::prev(_target_list->end())){
                // Deletion
                next_content = {current_pair->second->_content,
                                std::next(current_pair->second)->_content};
                next_iters = {current_pair->second,
                              std::next(current_pair->second)};
                //std::cout << "\tDeleting from first " << *(current_pair->second) << " " << *(std::next(current_pair->second)) << std::endl;
                delete_from_first(next_content, next_iters);
                // Substitution in string
                current_pair->first->_content = replacement;
                _target_list->erase(current_pair->second);
                // Insertion
                next_content = {replacement,
                                std::next(current_pair->second)->_content};
                if(_center_lists.count(next_content) == 0){
                    _center_lists.emplace(next_content,
                                          iter_pair_list());
                    _first_maps.emplace(next_content,
                                        iter_iter_pair_iter_map(10, iter_hasher));
                    _second_maps.emplace(next_content,
                                         iter_iter_pair_iter_map(10, iter_hasher));

                    _seen.emplace(next_content,
                                            iter_pair_set(10, iter_pair_hasher));
                    _masked.emplace(next_content,
                                            iter_pair_set(10, iter_pair_hasher));
                }
                _center_lists.at(next_content).push_back(next_iters);
                iter_pair_iter temp_next = std::prev(_center_lists.at(next_content).end());
                _first_maps.at(next_content)[next_iters.first] = temp_next;
                _second_maps.at(next_content)[next_iters.second] = temp_next;
                if(_second_maps.at(next_content).count(next_iters.first) == 0){
                    _seen.at(next_content).insert(next_iters);
                }else{
                    const iter_pair & previous_pair = *(_second_maps.at(next_content).at(next_iters.first));
                    if(_masked.at(next_content).count(previous_pair) == 0){
                        _masked.at(next_content).insert(next_iters);
                    }else{
                        _seen.at(next_content).insert(next_iters);
                    }
                }
            }else{
                // Substitution in string
                current_pair->first->_content = replacement;
                _target_list->erase(current_pair->second);
            }
            // Deletion from access maps
            _first_maps.at(xy).erase(current_pair->first);
            _second_maps.at(xy).erase(current_pair->second);
            _seen.at(xy).erase(*current_pair);
            _masked.at(xy).erase(*current_pair);
        }
        /*
        for(iter_pair & vw: _not_overlapping_anymore){
            int_pair vw_content = {vw.first->_content, vw.second->_content};
            _seen.at(vw_content).insert(vw);
            _masked.at(vw_content).erase(vw);
            // Flip the mask on the following
            while(_first_maps.at(vw_content).count(vw.second) != 0){
                iter_pair_iter & temp = _first_maps.at(vw_content).at(vw.second);
                if(_seen.at(vw_content).count(*temp) > 0){
                    _seen.at(vw_content).erase(*temp);
                    _masked.at(vw_content).insert(*temp);
                }else{
                    _seen.at(vw_content).insert(*temp);
                    _masked.at(vw_content).erase(*temp);
                }
                std::advance(vw.first, 1);
                std::advance(vw.second, 1);
                if(vw.second == _target_list->end()) break;
            }
        }
        */
        _center_lists.erase(xy);
        _first_maps.erase(xy);
        _second_maps.erase(xy);
        _seen.erase(xy);
        _masked.erase(xy);
        //std::cout << "DONE DELETING" << std::endl;
     }

    void lookup_forward_overlap(const iter_pair_iter & to_delete){
        iter & target = to_delete->first;
        if(std::next(std::next(target)) == _target_list->end()){
            return;
        }
        iter_pair next_pair = {std::next(target), std::next(std::next(target))};
        if(_masked.count({next_pair.first->_content, next_pair.second->_content}) > 0){
            if(_masked.at({next_pair.first->_content, next_pair.second->_content}).count(next_pair) > 0){
                _not_overlapping_anymore.push_back(next_pair);
            }
        }
    }

    void delete_from_first(const int_pair & content,
                           const iter_pair & target){
        if(_first_maps.count(content) == 0){
            std::cout << "Illegal content (first) " << content.first << " " << content.second << std::endl;
        }
        iter_pair_iter to_delete = _first_maps.at(content).at(target.first);
        //std::cout << "First: " << *(to_delete->first) << " " << *(to_delete->second) << std::endl;
        //lookup_forward_overlap(to_delete);
        _seen.at(content).erase(*to_delete);
        _masked.at(content).erase(*to_delete);
        _center_lists.at(content).erase(to_delete);
        _second_maps.at(content).erase(target.second);
        _first_maps.at(content).erase(target.first);
    }

    void delete_from_second(const int_pair & content,
                            const iter_pair & target){
        if(_second_maps.count(content) == 0){
            std::cout << "Illegal content (second) " << content.first << " " << content.second << std::endl;
        }
        iter_pair_iter to_delete = _second_maps.at(content).at(target.second);
        std::cout << "Second: " << *(to_delete->first) << " " << *(to_delete->second) << std::endl;
        _seen.at(content).erase(*to_delete);
        _masked.at(content).erase(*to_delete);
        _center_lists.at(content).erase(to_delete);
        _first_maps.at(content).erase(target.first);
        _second_maps.at(content).erase(target.second);
    }

    void print(const int_pair & target_pair){
        if(_center_lists.count(target_pair) == 0){
            std::cout << "Pair is not present" << std::endl;
            return;
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

    const overlap_access_map & get_seen() const{
        return _seen;
    }

    void print_center_lists(const int_string_map & translation_map) const{
        for(auto x : *(_target_list)){
            if(x._content >= 0){
                std::cout << translation_map.at(x._content) << " ";
            }else{
                std::cout << x._content << " ";
            }
        }std::cout << std::endl;
    }

};


#endif // MEM_SANDWICH_H
