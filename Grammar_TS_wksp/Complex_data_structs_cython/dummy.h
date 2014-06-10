#ifndef DUMMY_H
#define DUMMY_H

#include <vector>
#include <iostream>

template<typename T>
class Dummy{

private:
	std::pair<T, T> _content;

public:
	Dummy(){}

	void set_content(std::pair<T, T> content){
		_content = content;
	}

	void print_content(){
		std::cout << "Content left: " << _content.first << std::endl;
		std::cout << "Content right: " << _content.second << std::endl;
	}

};


#endif
