#ifndef SEQUITUR_H
#define SEQUITUR_H

#include<unordered_map>
#include<list>

#include"rule.h"
#include "name_generator.h"

template<typename T>
class Sequitur
{

typedef std::pair<T, Rule<T>*>                  pointing_c;
typedef std::list<pointing_c>                   any_gram;
typedef std::list<pointing_c>                               d_linked_list;
typedef typename d_linked_list::iterator                    it;
typedef std::unordered_map<T, std::list<int>>               hash_map_T_i;
typedef std::unordered_map<size_t,
        std::pair<it, int>>                                 hash_map_size_t_pair_it_i;
typedef std::unordered_map<size_t, T>                       hash_map_size_t_T;
typedef std::unordered_map<T, Rule<T>>                      hash_map_T_ruleT;
typedef typename hash_map_T_ruleT::iterator                 gram_it;

private:

    Name_generator<T>                       _name_gen;                                  // Name generator will create the new rule names
    T const *                               _input;                                     // Input sequence of T
    const int                               _n_elts;                                    // Number of elements in the input sequence
    int                                     _current_index;                             // Current index as one streams through the sequence
    int                                     _n_gram_size                = 2;            // Only digrams are considered as utiliy enforcement is terminal
    T                                       _root_id;                                   // Symbol of the stream's left hand side
    Rule<T> *                               _stream;                                    // Stream
    hash_map_T_ruleT                        _grammar;                                   // All the grammar
    hash_map_size_t_pair_it_i               _already_seen;                              // Already seen digrams
    hash_map_size_t_T                       _already_in_grammar;                        // Digrams that are already in the grammar
    std::list<T>                            _order_of_creation;                         // Used for grammar collapsing

public:

    // Transparent constructor
    Sequitur<T>(T* input, int n_elts, T root_id, Name_generator<T> name_gen) :
        _name_gen(name_gen),
        _input(input),
        _n_elts(n_elts),
        _current_index(0),
        _root_id(root_id)
    {
        _grammar.emplace(_root_id, Rule<T>(root_id, _grammar));
        _stream = &(_grammar.at(_root_id));
    }

    // Append next element of the input to _stream
    void append_next(){
        //std::cout << "Appending " << _input[_current_index] << std::endl;
        _stream->append_elt(_input[_current_index ++]);
    }

    void compute_next(){
        //_stream->light_print();
        size_t current_hash;
        if(_stream->get_RHS_size() < _n_gram_size) return;
        current_hash = _stream->get_terminal_hash(_n_gram_size);
        //std::cout << "Current hash " << (size_t) current_hash << std::endl;
        if(_already_in_grammar.count(current_hash) > 0){
            apply_old_rule(current_hash);
        }else if(_already_seen.count(current_hash) > 0){
            it start_previous = _already_seen[current_hash].first;
            std::advance(start_previous, _already_seen[current_hash].second);
            if((&(*(start_previous))) == (&(*(--(_stream->get_RHS().end()))))){
                // Three consecutive identical chars
                return;
            };
            create_new_rule(current_hash, _n_gram_size);
        }else{
            record_end(current_hash, _n_gram_size);
        }
    }

    void apply_old_rule(size_t current_hash){
        // The n_gram is already in the grammar
        // Replace it by the left hand side
        //std::cout << "Enforcing rule " << _already_in_grammar[current_hash] << std::endl;
        if(_stream->get_RHS_size() > 2){
            // Deletion of preterminal in already_seen
            it preterm_end = _stream->get_RHS().end();
            std::advance(preterm_end, -2);
            size_t hash_to_delete;
            _stream->get_backward_hash(preterm_end, 2, hash_to_delete);
            //std::cout << "Deleting already seen preterm hash " << hash_to_delete << std::endl;
            _already_seen.erase(hash_to_delete);
        }
        const T & LHS = _already_in_grammar[current_hash];
        Rule<T> & RHS = _grammar.at(LHS);
        RHS.apply(*_stream);
        //std::cout << "Next compute" << std::endl;
        compute_next(); // Check once more
    }

    void create_new_rule(size_t current_hash, int current_size){

        // The n_gram has already been seen in the sequence
        // Create a new rule
        T new_name = _name_gen.next_name();
        _order_of_creation.push_back(new_name);

        //std::cout << "Creating rule with name " << new_name << std::endl;

        it start = _already_seen[current_hash].first;
        int size = _already_seen[current_hash].second;
        if(size != current_size) return;
        it end = start;
        std::advance(end, size);

        //std::cout << "Start = " << start->first << " end= " << (--end)->first << std::endl;
        //++end;

        it first_term = _stream->get_RHS().end();
        std::advance(first_term, -current_size);
        bool consecutive = (&(*first_term)) == (&(*end));

        // Erasing already seen from table
        //std::cout << "Erasing current: " << current_hash << std::endl;
        _already_seen.erase(current_hash);
        size_t erase_forward_hash;
        size_t erase_backward_hash;
        size_t erase_terminal_hash;
        it forward_end = end;
        it terminal_end = _stream->get_RHS().end();
        std::advance(terminal_end, -(current_size));
        if(_stream->get_forward_hash(--forward_end, current_size, erase_forward_hash)){
            //std::cout << "Erasing forward: " << erase_forward_hash << std::endl;
            _already_seen.erase(erase_forward_hash);
        }
        if(_stream->get_backward_hash(start, current_size, erase_backward_hash)){
            //std::cout << "Erasing backward: " << erase_backward_hash << std::endl;
            _already_seen.erase(erase_backward_hash);
        }
        if(_stream->get_backward_hash(terminal_end, current_size, erase_terminal_hash)){
            //std::cout << "Erasing terminal: " << erase_terminal_hash << std::endl;
            _already_seen.erase(erase_terminal_hash);
        }

        _already_in_grammar.emplace(current_hash, new_name);
        bool at_begin = (start == _stream->get_RHS().begin());
        it end_first_backward;
        if(!at_begin){
            end_first_backward = start;
            --end_first_backward;
        }
        bool at_end = false;
        it begin_append = _stream->get_RHS().end();
        std::advance(begin_append, -current_size);
        if((&(*begin_append)) == (&(*end))){
            at_end = true;
        }

        _grammar.emplace(new_name, Rule<T>(new_name, _grammar));
        _grammar.at(new_name).build_from(*_stream, start, end);
        //std::cout << &(_grammar.at(new_name)) << std::endl;

        //std::cout << "Print before hashes" << std::endl;
        //_stream->print();

        // Update the new observed di or n_grams
        size_t first_backward_hash;
        it first_backward_hash_begin_pos;
        size_t first_forward_hash;
        it first_forward_hash_end_pos;
        size_t last_backward_hash;
        it last_backward_hash_begin_pos;
        int peek_size = _n_gram_size;
        //std::cout << "Peeking with size " << peek_size << std::endl;
        if(!at_begin){
            if(_stream->get_backward_hash(++end_first_backward, peek_size,
                                         first_backward_hash,
                                         first_backward_hash_begin_pos)){
                //std::cout << "Adding backward: " << first_backward_hash << std::endl;
                _already_seen[first_backward_hash] =
                        std::pair<it, int>(first_backward_hash_begin_pos,
                                          current_size);
            }
        }
        if(!consecutive){
            if((_stream->get_forward_hash(--end, peek_size,
                                        first_forward_hash,
                                        first_forward_hash_end_pos)
                ) && (!at_end)){
                //std::cout << "Adding forward: " << first_forward_hash << std::endl;
                _already_seen[first_forward_hash] =
                        std::pair<it, int>(end,
                                          current_size);
            }
        }
        last_backward_hash = _stream->get_terminal_hash(peek_size,
                                                       last_backward_hash_begin_pos);
        if(last_backward_hash == first_forward_hash){
            create_new_rule(last_backward_hash, current_size);
        }
        if(last_backward_hash == first_backward_hash){
            create_new_rule(last_backward_hash, current_size);
        }
        //std::cout << "Adding last backward: " << last_backward_hash << std::endl;
        _already_seen[last_backward_hash] = std::pair<it, int>(last_backward_hash_begin_pos,
                                                              current_size);
    }

    void record_end(size_t current_hash, int current_size){
        it start;
        _stream->get_terminal_hash(current_size, start);
        it end = start;
        std::advance(end, current_size);
        _already_seen[current_hash] = std::pair<it, int>(start, current_size);
    }

    void enforce_utility(){
        _order_of_creation.reverse();
        //std::cout << "ENFORCING UTILITY RULE" << std::endl;
        //std::cout << "Reverse order of creation" << std::endl;
        Rule<T> * current_rule;
        std::list<T> to_delete;
        for(T & x : _order_of_creation){
            //std::cout << x << std::endl;
            current_rule = &(_grammar.at(x));
            if(current_rule->get_ref_count() == 1){
                //std::cout << "Collapsing rule " << x << std::endl;
                current_rule->merge_with_last_ref();
                to_delete.push_back(x);
            }
        }
        for(T & x : to_delete){
            _grammar.erase(x);
        }
    }

    void print(){
        std::cout << "Grammar: " << std::endl;
        for(gram_it xy = _grammar.begin(); xy != _grammar.end(); ++xy){
            xy->second.print();
            std::cout << "|" <<std::endl;
        }
        std::cout << "Already seen map: " << std::endl;
        it start;
        it end;
        for(auto xy : _already_seen){
            start = xy.second.first;
            end = start;
            std::advance(end, xy.second.second - 1);
            std::cout << "Hash: " << (size_t) xy.first << " , first: " << start->first << " , second: " << end->first << std::endl;
        }
        std::cout << "Already in grammar map: " << std::endl;
        for(auto xy : _already_in_grammar){
            std::cout << "Hash: " << (size_t) xy.first << " , address: " << xy.second << std::endl;
        }
    }

    const hash_map_T_ruleT & get_grammar(){
        return _grammar;
    }

    const Rule<T> & get_stream(){
        return *_stream;
    }

};


#endif // SEQUITUR_H
