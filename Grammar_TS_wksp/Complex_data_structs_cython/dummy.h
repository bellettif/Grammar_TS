#ifndef DUMMY_H
#define DUMMY_H

#include <vector>
#include <iostream>

template<typename T>
class Dummy{

private:
	std::vector<T> &		_content;

public:
	Dummy(std::vector<T> & content):
		_content(content)
	{}

	void modify_content(int i,
						const T & value){
		_content[i] = value;
	}

	const std::vector<T> & get_content(){
		return _content;
	}

};


#endif
