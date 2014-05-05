#ifndef RULE_H
#define RULE_H

#include <list>

template<typename T>
class Rule{

typedef std::list<T>                    list_T;
typedef typename list_T::iterator       iter_list_T;
typedef std::pair<T, T>                 pair_T;

private:

    const T           _LHS;
    list_T            _RHS;

public:

    Rule(const T & LHS, const list_T & RHS):
        _LHS(LHS), _RHS(RHS){}

    Rule(const T & LHS, const pair_T & RHS):
        _LHS(LHS){
        _RHS.push_back(RHS.first);
        _RHS.push_back(RHS.second);
    }

    void apply(std::list<T> & input_string,
               iter_list_T & first, iter_list_T & last){
        /*
        std::cout << "Applying rule " << _LHS << " to "
                  << *first << "-" << *last << std::endl;
        */
        iter_list_T saved_first;
        bool begin;
        if(first == input_string.begin()){
            begin = true;
        }else{
            begin = false;
            saved_first = first;
            -- saved_first;
        }
        input_string.erase(first, ++last);
        if(begin){
            first = input_string.insert(input_string.begin(), _LHS);
        }else{
            first = input_string.insert(++ saved_first, _LHS);
        }
        last = first;
    }

    void merge(list_T & other_rule,
               iter_list_T start){
        bool begin = false;
        iter_list_T pre;
        if(start == other_rule.begin()){
            begin = true;
        }else{
            pre = start;
            -- pre;
        }
        other_rule.erase(start);
        if(begin){
            pre = other_rule.begin();
            other_rule.insert(pre, _RHS.begin(), _RHS.end());
        }else{
            ++ pre;
            other_rule.insert(pre, _RHS.begin(), _RHS.end());
        }
    }

    list_T & get_rhs(){
        return _RHS;
    }

    T get_lhs(){
        return _LHS;
    }

    list_T * get_rhs_pt(){
        return & _RHS;
    }

    void print() const{
        std::cout << "LHS: " << _LHS << " -> ";
        for(auto x : _RHS){
            std::cout << x << " ,";
        }std::cout << std::endl;
    }





};

#endif // RULE_H
