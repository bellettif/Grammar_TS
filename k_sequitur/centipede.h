#ifndef CENTIPEDE_H
#define CENTIPEDE_H

#include <unordered_map>
#include <list>

/**
 *  Core memory data structure of the k-Sequitur algorithm.
 *      See master's thesis for a complete description.
 *      This data structure keeps track of all occurences of a pair
 *      in the input string, allowing retrieval of the pointers towards
 *      the pre and suffixes of the corresponding elements in constant access time.
 */
template<typename T>
class Centipede{

typedef std::list<T>                                    T_list;
typedef typename T_list::iterator                       iter;
typedef std::pair<T, T>                                 T_pair;
typedef std::pair<iter, iter>                           iter_pair;
typedef std::list<iter>                                 iter_list;
typedef std::list<iter_pair>                            iter_pair_list;
typedef typename iter_pair_list::iterator               iter_pair_list_iter;


typedef std::function<size_t(iter)>                     iter_hash;
typedef std::unordered_map<iter,
                           iter_pair_list_iter,
                           iter_hash>                   mem_mask;

/**
*   Hashing function for iterators that considers the position in the input string.
*       In order to have a non ambiguous position index, the memory address of the element
*       is considered. This can become faulty in case of memory reallocation and is corrected
*       int the proba-Sequitur version of the algorithm.
*/
iter_hash iterator_hasher = [](const iter & x){
    return (size_t) &(*x);
};


private:
/**
* @brief _first_hash_set
*   Lateral constant access time structure (hash map based
*       on the position the first element of the pair points
*       to in the input string).
*/
    mem_mask                _first_hash_set;

/**
 * @brief _core
 *  Double linked list of pairs of iterators towards the input string.
 *      It allows a left to right iteration through the previous occurences of
 *      a given pair of symbols.
 */
    iter_pair_list          _core;                  // Core list

/**
 * @brief _second_hash_set
 *  Same as _first_hash_set but based on the position of the pointed address of
 *      the second element.
 */
    mem_mask                _second_hash_set;       // Lateral constant access


public:
    /**
     * @brief Centipede
     *  Default constructor, hashmaps need to be initialized explicitely
     *      for them to use the hash function defined above for pairs of
     *      iterators.
     */
    Centipede():
        _first_hash_set(128, iterator_hasher),
        _second_hash_set(128, iterator_hasher){}

    /**
     * @brief pop_front
     * @return
     */
    iter_pair pop_front(){
        iter_pair front = _core.front();
        _core.pop_front();
        _first_hash_set.erase(front.first);
        _second_hash_set.erase(front.second);
        return front;
    }

    void push_back(const iter_pair & xy){
        _core.push_back(xy);
        iter_pair_list_iter end = _core.end();
        --end;
        _first_hash_set.emplace(xy.first, end);
        _second_hash_set.emplace(xy.second, end);
    }

    bool delete_ref_first(const iter & x){
        if(_first_hash_set.count(x) == 0) return false;
        _second_hash_set.erase(_first_hash_set.at(x)->second);
        _core.erase(_first_hash_set.at(x));
        _first_hash_set.erase(x);
        return true;
    }

    bool delete_ref_second(const iter & x){
        if(_second_hash_set.count(x) == 0) return false;
        _first_hash_set.erase(_second_hash_set.at(x)->first);
        _core.erase(_second_hash_set.at(x));
        _second_hash_set.erase(x);
        return true;
    }

    int size() const{
        return _first_hash_set.size();
    }

    void print() const{
        for(iter_pair x : _core){
            std::cout << "\tPair: " << *(x.first);
            std::cout << "(" << &(*x.first) << ")";
            std::cout << "-" << *(x.second);
            std::cout << "(" << &(*x.second) << ")";
            std::cout << " first: " << *(_first_hash_set.at(x.first)->first);
            std::cout << " second: " << *(_second_hash_set.at(x.second)->second);
            std::cout << std::endl;
        }
    }


};



#endif // CENTIPEDE_H
