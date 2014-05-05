#ifndef NAME_GENERATOR_H
#define NAME_GENERATOR_H

#include<vector>
#include<string>

namespace alphabets{
    const std::vector<char> ascii_lower = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    const std::vector<char> ascii_upper = {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
    const std::vector<std::string> ascii_lower_string = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"};
    const std::vector<std::string> ascii_upper_string = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"};
}


template<typename T>
class Name_generator
{

private:
    const std::vector<T>        _alphabet;
    int                         _current_index;
    const int                   _alphabetsize;

public:
    Name_generator<T>(const std::vector<T> & alphabet):
        _alphabet(alphabet), _current_index(0),
        _alphabetsize(alphabet.size())
    {}

    T next_name(){
        return _alphabet[_current_index++];
    }
};


template<>
class Name_generator<std::string>
{

typedef std::string         T;

private:
    const std::vector<T>        _alphabet;
    int                         _current_index;
    const int                   _alphabetsize;
    T                           _prefix = "";

public:
    Name_generator<T>(const std::vector<T> & alphabet):
        _alphabet(alphabet), _current_index(0),
        _alphabetsize(alphabet.size())
    {}

    T next_name(){
        if(_current_index == _alphabetsize){
            _current_index = 0;
            _prefix += "_";
        }
        return _prefix + _alphabet[_current_index++];
    }

};

template<>
class Name_generator<int>
{

typedef int                     T;

private:
    const std::vector<T>        _alphabet;
    int                         _current_index;
    const int                   _alphabetsize;
    const bool                  _empty_alphabet;

public:
    Name_generator<T>(const std::vector<T> & alphabet):
        _alphabet(alphabet), _current_index(0),
        _alphabetsize(alphabet.size()),
        _empty_alphabet(false)
    {}

    Name_generator<T>(int root = 0):
        _current_index(root),
        _alphabetsize(0),
        _empty_alphabet(true)
    {}

    T next_name(){
        if(_empty_alphabet){
            return (--_current_index);
        }else{
            return _alphabet[_current_index++];
        }
    }
};



#endif // NAME_GENERATOR_H
