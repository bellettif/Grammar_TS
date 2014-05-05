#include<unordered_map>
#include<vector>
#include<tuple>
#include<boost/functional/hash.hpp>


template<typename T>
class Entropy_measure{

private:
	std::unordered_map<T, int> 			_count_map;
	int 								_total_count;

public:
	Entropy_measure():
		_total_count(0){}

	Entropy_measure(const std::vector<T> & input_data):
		_total_count(0)
	{
		for(const T & x : input_data){
			++ _total_count;
			if(_count_map.count(x) == 0){
				_count_map[x] = 1;
			}else{
				++ _count_map[x];
			}
		}
	}

	void reset(const std::vector<T> & input_data){
		_total_count = 0;
		_count_map.clear();
		for(const T & x : input_data){
			++ _total_count;
			if(_count_map.count(x) == 0){
				_count_map[x] = 1;
			}else{
				++ _count_map[x];
			}
		}
	}

	void append_obs(const T & datum){
		if(_count_map.count(datum) == 0){
			_count_map[datum] = 1;
		}else{
			++ _count_map[datum];
		}
	}

	void append_obs(const std::vector<T> & data){
		for(const T & x : data){
			append_obs(x);
		}
	}

	void show_counts(){
		for(auto xy : _count_map){
			std::cout << "Key: " << xy.first << " Count:" << xy.second << std::endl;
		}
	}

	double compute_entropy(){
		double result = 0;
		double temp;
		for(auto xy : _count_map){
			temp = ((double) xy.second) / ((double) _total_count);
			result += std::log2(temp) * temp;
		}
		return -result;
	}

	double compute_gini(){
		double result = 0;
		double temp;
		for(auto xy : _count_map){
			temp = ((double) xy.second) / ((double) _total_count);
			result += (1.0 - temp) * temp;
		}
		return result;
	}
};



template<typename T>
class Entropy_measure<std::vector<T>>{

typedef std::vector<T>											vect;
typedef std::function<std::size_t(const vect &)>				hash_func_type;

hash_func_type boost_hash = [](const vect & sequence){
	std::size_t hash = 0;
	boost::hash_range(hash, sequence.begin(), sequence.end());
	return hash;
};

private:
	std::unordered_map<vect, int,
					   hash_func_type> 							_count_map;
	int 														_total_count;

public:
	Entropy_measure(int size = 100):
		_count_map(size, boost_hash),
		_total_count(0){}

	Entropy_measure(const std::vector<vect> & input_data, int size = 100):
		_count_map(size, boost_hash),
		_total_count(0)
	{
		for(const vect & x : input_data){
			++ _total_count;
			if(_count_map.count(x) == 0){
				_count_map[x] = 1;
			}else{
				++ _count_map[x];
			}
		}
	}

	void reset(const std::vector<vect> & input_data){
		_total_count = 0;
		_count_map.clear();
		for(const T & x : input_data){
			++ _total_count;
			if(_count_map.count(x) == 0){
				_count_map[x] = 1;
			}else{
				++ _count_map[x];
			}
		}
	}

	void append_obs(const vect & datum){
		++ _total_count;
		if(_count_map.count(datum) == 0){
			_count_map[datum] = 1;
		}else{
			++ _count_map[datum];
		}
	}

	void append_obs(const std::vector<vect> & data){
		for(const vect & x : data){
			append_obs(x);
		}
	}

	void show_counts(){
		for(auto xy : _count_map){
			std::cout << "Key: ";
			for(auto x : xy.first){
				std::cout << x;
			}
			std::cout << " Count: ";
			std::cout << xy.second << std::endl;
		}
	}

	double compute_entropy(){
		double result = 0;
		double temp;
		for(auto xy : _count_map){
			temp = ((double) xy.second) / ((double) _total_count);
			result += std::log2(temp) * temp;
		}
		return -result;
	}

	double compute_gini(){
		double result = 0;
		double temp;
		for(auto xy : _count_map){
			temp = ((double) xy.second) / ((double) _total_count);
			result += (1.0 - temp) * temp;
		}
		return result;
	}
};
