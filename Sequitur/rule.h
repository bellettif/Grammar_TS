#ifndef RULE_H
#define RULE_H

#include<unordered_map>
#include<list>
#include<unordered_set>

namespace std{
    static std::string to_string(std::string value){
        return value;
    }
}


template<typename T>
class Rule
{

typedef std::pair<T, Rule<T>*>                                          pointing_c;
typedef std::list<pointing_c>                                           d_linked_list;
typedef typename d_linked_list::iterator                                it;
typedef std::function<size_t(it)>                                       it_hash_func;
typedef std::unordered_set<it, it_hash_func>                            it_unordered_set;
typedef std::unordered_map<T, it_unordered_set>                         hash_map_T_it;
typedef std::unordered_map<T, Rule<T>>                                  hash_map_T_ruleT;
typedef typename hash_map_T_ruleT::iterator                             gram_it;

// Custom hash function for iterators based on the pointer values
it_hash_func my_hash = [](const it & x){
    return (size_t) &(*(x));
};

private:
    const T                     _LHS;               // Rule left hand side
    int                         _RHS_length;        // Rule right hand side length
    d_linked_list               _RHS;               // Rule right hand side
    hash_map_T_it               _refs;              // Refs to the rule
    int                         _ref_count;         // Ref count to the rule (for utility enforcement)
    hash_map_T_ruleT &          _gram;              // Reference to grammar (needed to collapse the grammar)

public:

    // Transparent constructor
    Rule<T>(T LHS, hash_map_T_ruleT & gram):
        _LHS(LHS), _RHS_length(0), _gram(gram), _ref_count(0){}

    // Create a new rule, extracted from symbols in the parent between start and end
    void build_from(Rule & parent,
                    it start,
                    it end)
    {

        bool at_begin = (start == parent._RHS.begin());
        it insertion_point = start;
        if(at_begin){
            insertion_point = parent._RHS.begin();
        }else{
            std::advance(insertion_point, -1);
        }

        // Insert first occurence of rule in parent into new rule
        // Will define RHS by transfering the elements from the parents
        for(it x = start; x != end; ++x){
            if(x->second != 0){
                x->second->change_ref_count(parent._LHS, _LHS, x);
            }
        }
        _RHS.splice(_RHS.begin(), parent._RHS, start, end);
        _RHS_length = _RHS.size();

        // Delete second occurence in parent (from the end)
        // Erase will trigger the destruction of the elements removed from the list
        int parent_RHS_length = parent.get_RHS().size();
        int back_track_index = parent_RHS_length - (_RHS_length);
        it cut_end_it = parent._RHS.begin(); // For first insert (second occurence, at the end)
        std::advance(cut_end_it, back_track_index);
        for(it x = cut_end_it; x != parent._RHS.end(); ++x){
            if(x->second != 0){
                x->second->dec_ref_count(parent._LHS, x);
            }
        }
        parent._RHS.erase(cut_end_it, parent._RHS.end());

        // Insert LHS char in parent
        parent._RHS.push_back(pointing_c(_LHS, this));
        if(at_begin){
            parent._RHS.insert(parent._RHS.begin(), pointing_c(_LHS, this));
        }else{
            parent._RHS.insert(++insertion_point, pointing_c(_LHS, this));
        }

        // Add reference from rule to second occurence
        // Create reference from new rule to parent
        _ref_count = 2;
        _refs.emplace(parent.get_LHS(), it_unordered_set(10, my_hash));
        if(at_begin){
            _refs[parent.get_LHS()].insert(parent._RHS.begin());
        }else{
            _refs[parent.get_LHS()].insert(--insertion_point);
        }
        _refs[parent.get_LHS()].insert(--(parent._RHS.end()));

    }

    // Apply rule to the two last symbols of the parent's right hand side
    void apply(Rule & parent){
        //std::cout << "Applying rule" << std::endl;
        int parent_RHS_length = parent.get_RHS().size();
        int back_track_index = parent_RHS_length - (_RHS_length);
        it cut_end_it = parent._RHS.begin(); // For first insert (second occurence, at the end)
        std::advance(cut_end_it, back_track_index);
        for(it x = cut_end_it; x != parent._RHS.end(); ++x){
            if(x->second != 0){
                x->second->dec_ref_count(parent._LHS, x);
            }
        }

        int rhs_size = _RHS.size();
        it end_it = parent._RHS.end();
        std::advance(end_it, -rhs_size);
        parent._RHS.erase(end_it, parent._RHS.end());
        //std::cout << "Done erasing" << std::endl;
        parent._RHS.push_back(pointing_c(_LHS, this));
        if(_refs.count(parent.get_LHS()) == 0){
            _refs.emplace(parent.get_LHS(), it_unordered_set(10, my_hash));
        }
        _refs[parent.get_LHS()].insert(--(parent._RHS.end()));
        ++_ref_count;
        //std::cout << "Done applying rule" << std::endl;
    }

    // Append an element to the right hand side (used for the main symbol stream)
    void append_elt(T elt){
        _RHS.push_back(pointing_c(elt, 0));
    }

    // Simple getter !!! Reference passing
    d_linked_list & get_RHS(){
        return _RHS;
    }

    // Const preserving getter
    const d_linked_list & get_RHS() const{
        return _RHS;
    }

    // Simple getter, const preserving
    int get_RHS_size() const{
        return _RHS.size();
    }

    // Simple getter, const preserving
    const T & get_LHS() const{
        return _LHS;
    }

    // Get references to the rule
    const hash_map_T_it & get_refs() const{
        return _refs;
    }

    // Get ref count of the rule
    int get_ref_count() const{
        return _ref_count;
    }

    // Collapse the rule with the last rule that references it
    void merge_with_last_ref(){
        it last_ref;
        T LHS;
        for(auto xy : _refs){
            last_ref = *(xy.second.begin());
            LHS = xy.first;
            break;
        }
        Rule<T> & to_merge_with = _gram.at(LHS);
        to_merge_with._RHS.splice(last_ref, _RHS);
        to_merge_with._RHS.erase(last_ref);
    }

    // Decrement reference count
    void dec_ref_count(T source_LHS, it origin){
        /*
        std::cout << "Decrementing ref count of " << _LHS
                  << " (" << this << ")" << std::endl;
        */
        _refs[source_LHS].erase(origin);
        if(_refs[source_LHS].size() == 0){
            _refs.erase(source_LHS);
        }
        --_ref_count;
    }

    // Change referencing
    void change_ref_count(T source_LHS, T new_LHS, it origin){
        /*
        std::cout << "Changin ref count from " << source_LHS << " to "
                  << new_LHS << " at " << _LHS << " (" << this << ")" << std::endl;
        */
        _refs[source_LHS].erase(origin);
        if(_refs[source_LHS].size() == 0){
            _refs.erase(source_LHS);
        }
        //std::cout << "Erase done" << std::endl;
        if(_refs.count(new_LHS) == 0){
            _refs.emplace(new_LHS, it_unordered_set(10, my_hash));
        }
        _refs[new_LHS].insert(origin);
        //std::cout << "Insert done" << std::endl;
    }

    // Simple printer, all information
    void print(){
        std::cout << _LHS << "|" << this << " ref count " << _ref_count;
        for(auto xy : _refs){
            std::cout << " refed in " << xy.first;
            for(auto z : xy.second){
                std::cout << " at " << &(*z);
            }
        }
        std::cout << " -> ";
        for(it x = _RHS.begin(); x != _RHS.end(); ++x){
            std::cout << x->first << "|" << x->second << "(" << &(*x) << ") " ;
        }std::cout << "\n";
    }

    // Simple printer, little information
    void light_print(){
        std::cout << _LHS << " -> ";
        for(it x = _RHS.begin(); x != _RHS.end(); ++x){
            std::cout << x->first;
        }std::cout << "\n";
    }

    // Get hash of the current_size last elements
    size_t get_terminal_hash(int current_size){
        std::string concat;
        it back_it = _RHS.end();
        --back_it;
        for(int i = 0; i < current_size; ++i){
            concat.insert(0, std::to_string((back_it--)->first));
        }
        return std::hash<std::string>()(concat);
    }

    // Get terminal hash and store last state of iterator in begin_pos
    size_t get_terminal_hash(int current_size, it & begin_pos){
        std::string concat;
        it back_it = _RHS.end();
        --back_it;
        for(int i = 0; i < current_size; ++i){
            concat.insert(0, std::to_string((back_it--)->first));
        }
        begin_pos = ++back_it;
        return std::hash<std::string>()(concat);
    }

    // Get forward hash (forwarded of current_size elements)
    bool get_forward_hash(it pos, int current_size, size_t & hash){
        std::string concat;
        for(int i = 0; i < current_size; ++i){
            if(pos == _RHS.end()) return false;
            concat += std::to_string((pos++)->first);
        }
        hash = std::hash<std::string>()(concat);
        return true;
    }

    // Get forward hash and store last position of iterator in end_pos
    bool get_forward_hash(it pos, int current_size,
                          size_t & hash, it & end_pos){
        std::string concat;
        for(int i = 0; i < current_size; ++i){
            if(pos == _RHS.end()) return false;
            concat += std::to_string((pos++)->first);
        }
        hash = std::hash<std::string>()(concat);
        end_pos = pos;
        return true;
    }

    // Get backward hash (backwarded of current_size elements)
    bool get_backward_hash(it pos, int current_size,
                           size_t & hash){
        std::string concat;
        bool hit_return = false;
        for(int i = 0; i < current_size; ++i){
            if(hit_return) return false;
            hit_return = (pos == _RHS.begin());
            concat.insert(0, std::to_string((pos--)->first));
        }
        hash = std::hash<std::string>()(concat);
        return true;
    }

    // Get backward hash and store last position of iterator in begin_pos
    bool get_backward_hash(it pos, int current_size,
                           size_t & hash, it & begin_pos){
        std::string concat;
        bool hit_return = false;
        for(int i = 0; i < current_size; ++i){
            if(hit_return) return false;
            hit_return = (pos == _RHS.begin());
            concat.insert(0, std::to_string((pos--)->first));
        }
        hash = std::hash<std::string>()(concat);
        begin_pos = ++pos;
        return true;
    }


};


#endif // RULE_H
