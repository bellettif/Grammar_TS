#ifndef RULE_H
#define RULE_H

#include <string>
#include <vector>
#include <list>
#include <unordered_map>

using namespace std;

class rule
{

private:
    unordered_map<int, list<int>>       _list_of_begin_positions;
    double                              _importance;
    int                                 _rule_name;

public:
    rule();

};

#endif // RULE_H
