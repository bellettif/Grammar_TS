#ifndef PARSE_TREE_H
#define PARSE_TREE_H

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

};


#endif // PARSE_TREE_H
