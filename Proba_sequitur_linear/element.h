#ifndef ELEMENT_H
#define ELEMENT_H

#include <list>

struct Element{

    const int                               _seq_index;
    const int                               _word_index;
    bool                                    _has_prev = false;
    std::list<Element>::iterator            _prev;
    bool                                    _has_next = false;
    std::list<Element>::iterator            _next;
    int                                     _content;

public:
    std::list<Element>::iterator            _iter;

public:
    Element(const int & seq_index,
            const int & word_index,
            const int & content):
        _seq_index(seq_index),
        _word_index(word_index),
        _content(content)
    {}


};

std::ostream & operator<< (std::ostream & out, const Element & e){
    /*
    out << "(";
    if (e._has_prev){
        out << &(*e._prev) << ", ";
    }else{
        out << "no prev, ";
    }
    out << e._seq_index << ", "
        << e._word_index << ", "
        << &(*e._iter) << ", "
        << e._content << ", ";
    if(e._has_next){
        out << &(*e._next);
    }else{
        out << "no next";
    }
    out << ")";
    */
    out << " " << e._content << " ";
    return out;
}

#endif // ELEMENT_H
