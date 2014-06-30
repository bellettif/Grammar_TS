#ifndef COUNTER_H
#define COUNTER_H

#include <boost/functional/hash.hpp>

#include "centipede.h"

/**
*   Believe it or not, standard lib doesn't have a to_string
*       taking strings as values. This is necessary for class templating
*       so T can also be of type std::string.
*/
namespace std{
    static std::string to_string(std::string value){
        return value;
    }
}


/**
 *  Class that wraps a hashmap of pairs as keys to centipedes
 */
template<typename T>
class Counter{

typedef std::list<T>                                    T_list;
typedef typename T_list::iterator                       iter;
typedef std::pair<T, T>                                 T_pair;
typedef std::pair<iter, iter>                           iter_pair;
typedef std::list<iter>                                 iter_list;
typedef std::list<iter_pair>                            iter_pair_list;
typedef typename iter_list::iterator                    iter_list_iter;

/**
 *  Define a hashing function for pairs of symbols
 */
typedef std::function<size_t(T_pair)>                   T_pair_hash;
public: T_pair_hash T_pair_hasher = [&](const T_pair & pair){
    boost::hash<std::pair<T, T>> hasher;
    return hasher(std::pair<T, T>(pair.first, pair.second));
};

/**
 * @brief Mem_map
 *  This type maps pairs of symbols to the corresponding
 *      book-keeping centipede unit
 */
typedef std::unordered_map<T_pair,
                           Centipede<T>,
                           T_pair_hash>                 Mem_map;



private:
    /**
     * @brief _memory_map
     *  Main element of the memory that maps pairs of symbols onto
     *      centipedes.
     */
    Mem_map         _memory_map;


public:
    /**
     * @brief Counter
     *  Default constructor. Hashmap needs to be explecitely defined
     *      so it uses the proper hashcode function.
     */
    Counter():
        _memory_map(128, T_pair_hasher){}


    /**
     * @brief add_ref
     * @param ref_pair
     *  Add a reference pointing the positions of the elements of the pair
     *      in the centipeded corresponding to the pair of values the pair
     *      of pointers points towards.
     */
    void add_ref(const iter_pair & ref_pair){
        T_pair value_pair (*(ref_pair.first), *(ref_pair.second));
        if(_memory_map.count(value_pair) == 0){
            _memory_map.emplace(value_pair, Centipede<T>());
        }
        _memory_map.at(value_pair).push_back(ref_pair);
    }


    int get_ref_count(const T_pair & value_pair) const{
        if(_memory_map.count(value_pair) == 0){
            return 0;
        }else{
            return _memory_map.at(value_pair).size();
        }
    }


    iter_pair pop_front(const T_pair & values){
        return _memory_map.at(values).pop_front();
    }


    // Looks back and ahead
    void delete_ref(const iter_pair & ref_pair,
                    T_list & input,
                    const iter & stop){
        if(ref_pair.first != input.begin()){
            iter_pair pre_iters(ref_pair.first, ref_pair.first);
            -- pre_iters.first;
            T_pair pre_values(*(pre_iters.first), *(pre_iters.second));
            _memory_map.at(pre_values).delete_ref_second(pre_iters.second);
        }
        iter pre_end = stop;
        -- pre_end;
        if(ref_pair.second != pre_end){
            iter_pair post_iters (ref_pair.second, ref_pair.second);
            ++ post_iters.second;
            T_pair post_values(*(post_iters.first), *(post_iters.second));
            _memory_map.at(post_values).delete_ref_first(post_iters.first);
        }
    }


    // Looks back and ahead
    void create_ref(const iter_pair & ref_pair,
                    T_list & input,
                    iter stop){
        if(ref_pair.first != input.begin()){
            iter_pair pre_iters (ref_pair.first, ref_pair.first);
            -- pre_iters.first;
            T_pair pre_values (*(pre_iters.first), *(pre_iters.second));
            if(_memory_map.count(pre_values) == 0){
                _memory_map.emplace(pre_values, Centipede<T>());
            }
            _memory_map.at(pre_values).push_back(pre_iters);
        }
        iter pre_end = stop;
        -- pre_end;
        if(ref_pair.second != pre_end){
            iter_pair post_iters (ref_pair.second, ref_pair.second);
            ++ post_iters.second;
            T_pair post_values (*(post_iters.first), *(post_iters.second));
            if(_memory_map.count(post_values) == 0){
                _memory_map.emplace(post_values, Centipede<T>());
            }
            _memory_map.at(post_values).push_back(post_iters);
        }
    }


    void print() const{
        for(auto xy : _memory_map){
            std::cout << "Key (" << get_ref_count(xy.first) << ") :"
                      << xy.first.first
                      << "_" << xy.first.second << std::endl;
            xy.second.print();
        }
    }

};


#endif // COUNTER_H
