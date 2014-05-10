#ifndef INSIDE_PROBA_H
#define INSIDE_PROBA_H

#include <vector>

template<typename T>
class Inside_proba{

typedef std::vector<T>      T_vect;

private:
    const int               _start;
    const int               _end;
    const T_vect *          _input;

public:
    Inside_proba(const int & start,
                 const int & end,
                 const T_vect * input):
        _start(start),
        _end(end),
        _input(input){}



};


#endif // INSIDE_PROBA_H
