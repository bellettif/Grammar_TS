#ifndef DEPENDENCY_TREE_H
#define DEPENDENCY_TREE_H

#include<unordered_set>

#include"rule.h"

template<typename T>
class Dependency_tree{

typedef std::unordered_map<T, Rule<T>>                      hash_map_T_ruleT;
typedef typename hash_map_T_ruleT::iterator                 gram_it;
typedef std::pair<T, int>                                   counting_T;
typedef std::pair<counting_T, std::list<T>>                 Node;

private:
    hash_map_T_ruleT &          _grammar;
    Rule<T> &                   _stream;
    Node                        _root;
    std::list<Node>             _leaves;

public:
    Dependency_tree<T>(Rule<T>* stream, hash_map_T_ruleT & grammar):
        _grammar(grammar), _stream(*stream),
        _root(Node(counting_T(_stream.get_LHS(), 0), std::list<T>()))
    {}

    void build_initial_tree(){
        build_dfs(_root);
    }

    void build_dfs(Node & node){
        std::list<Node> to_do;
        T lhs;
        for(auto x : _grammar.at(node.first.first).get_RHS()){
            if(x.second != 0){
                ++node.first.second;
                lhs = x.second->get_LHS();
                node.second.push_back(lhs);
                to_do.push_back(Node());
            }
        }
        if(node.second.size() == 0){
            _leaves.push_back(node);
        }else{
            for(auto x : to_do){
                build_dfs(x);
            }
        }
    }

    void print_node(Node node){
        std::cout << node.first.first << std::endl;
    }

    void print_bfs(){
        print_node(_root);
        std::list<Node> to_do = _root.second;
        std::list<Node> next_to_do;
        while(to_do.size() > 0){
            for(auto x : to_do){
                print_node(x); std::cout << " | " << std::endl;
                next_to_do.insert(x.second);
            }
            to_do.swap(next_to_do);
        }
    }


};


#endif // DEPENDENCY_TREE_H
