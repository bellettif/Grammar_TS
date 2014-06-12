#ifndef ELEMENT_H
#define ELEMENT_H

#include <list>

struct Element{

    const int                               _seq_index;
    const int                               _word_index;
    int                                     _content;

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
    out << "(";
    out << e._seq_index << ", "
        << e._word_index << ", "
        << & e << ", "
        << e._content << ", ";
    out << ")";
    //out << " " << e._content << " ";
    return out;
}

#endif // ELEMENT_H
