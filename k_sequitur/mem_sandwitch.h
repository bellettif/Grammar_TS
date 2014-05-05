#ifndef MEM_SANDWITCH_H
#define MEM_SANDWITCH_H

#include <unordered_map>
#include <list>


template<typename T>
class Mem_sandwitch{

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

iter_hash iterator_hasher = [](const iter & x){
    return (size_t) &(*x);
};


private:
    mem_mask                _first_hash_set;
    mem_mask                _second_hash_set;
    iter_pair_list          _core;


public:
    Mem_sandwitch():
        _first_hash_set(128, iterator_hasher),
        _second_hash_set(128, iterator_hasher){}

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



#endif // MEM_SANDWITCH_H
