#include "stochastic_rule.h"

#include "scfg.h"

std::list<T> Stochastic_rule::complete_derivation(SCFG & grammar){
    std::list<T> result;
    bool is_terminal;
    derivation_result temp = derive(is_terminal);
    if(is_terminal){
        result.push_back(temp.first);
        return result;
    }else{
        std::list<T> temp_left = grammar.get_rule(temp.second.second).complete_derivation(grammar);
        result.insert(result.begin(),
                      temp_left.begin(),
                      temp_left.end());
        std::list<T> temp_right = grammar.get_rule(temp.second.first).complete_derivation(grammar);
        result.insert(result.begin(),
                      temp_right.begin(),
                      temp_right.end());
        return result;
    }
}
