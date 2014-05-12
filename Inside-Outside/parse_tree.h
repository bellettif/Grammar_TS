#ifndef PARSE_TREE_H
#define PARSE_TREE_H

#include <list>

#include "stochastic_rule.h"


template<typename T>
class Parse_tree{

typedef Stochastic_rule<T>      SGrammar_T;

private:
    int             _lhs;
    bool            _isterm;
    T               _term;
    int             _left_rule_name;
    Parse_tree *    _left_child = 0;
    int             _right_rule_name;
    Parse_tree *    _right_child = 0;

public:
    Parse_tree(int root):
        _lhs(root),
        _isterm(false){}

    Parse_tree(int root,
               Parse_tree * left_child,
               Parse_tree * right_child):
        _lhs(root),
        _left_rule_name(_left_child->lhs),
        _left_child(left_child),
        _right_rule_name(_right_child->lhs),
        _right_child(right_child){}

    Parse_tree(int root,
               const T & term):
        _lhs(root),
        _term(term){}

    ~Parse_tree(){
        delete _left_child;
        delete _right_child;
    }

    void set_terminal(T term){
        _isterm = true;
        _term = term;
        _left_child = 0;
        _right_child = 0;
    }

    void set_non_terminal(Parse_tree<T> * left, Parse_tree<T> * right){
        _isterm = false;
        _left_child = left;
        _left_rule_name = _left_child->_lhs;
        _right_child = right;
        _right_rule_name = _right_child->_lhs;
    }

    std::string to_string(){
        std::string result;
        if(_isterm){
            result += _term;
        }else{
            result += _left_child + " " + _right_child;
        }
        return result;
    }

    void print_all_tree(){
        std::list<Parse_tree<T> *>  current;
        std::list<Parse_tree<T> *>  next;
        current.push_back(this);
        std::cout << _lhs << std::endl;
        int counter;
        while(!current.empty()){
            counter = 1;
            for(Parse_tree<T>* x : current){
                if(counter++ < current.size()){
                    std::cout << *x << " | ";
                }else{
                    std::cout << *x;
                }
                if(!(x->is_term())){
                    next.push_back(x->get_left_child());
                    next.push_back(x->get_right_child());
                }
            }std::cout << std::endl;
            current.swap(next);
            next.clear();
        }
    }

    bool is_term() const{
        return _isterm;
    }

    T get_term() const{
        return _term;
    }

    int get_left_symbol() const{
        return _left_child->_lhs;
    }

    Parse_tree<T>* get_left_child() const{
        return _left_child;
    }

    int get_right_symbol() const{
        return _right_child->_lhs;
    }

    Parse_tree<T>* get_right_child() const{
        return _right_child;
    }

};

template<typename T>
std::ostream& operator<<(std::ostream& out, const Parse_tree<T>& p){
    if(p.is_term()){
        return out << p.get_term();
    }else{
        return out << p.get_left_symbol() << " " << p.get_right_symbol();
    }
}


#endif // PARSE_TREE_H
