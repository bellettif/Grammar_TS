#ifndef ELEMENT_H
#define ELEMENT_H

#include <list>

struct Element{

    const int                               _seq_index;
    const int                               _word_index;
    std::list<Element>::iterator            _iter;
    bool                                    _has_prev = false;
    std::list<Element>::iterator            _prev;
    bool                                    _has_next = false;
    std::list<Element>::iterator            _next;
    const int                               _content;

    Element(const int & seq_index,
            const int & word_index,
            const int & content):
        _seq_index(seq_index),
        _word_index(word_index),
        _content(content)
    {}


};

#endif // ELEMENT_H
