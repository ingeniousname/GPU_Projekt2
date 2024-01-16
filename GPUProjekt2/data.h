#pragma once
#include <vector>
#include "solving_strategy.h"
#include <memory>

class StringsData
{
public:
	bool transposed;
	unsigned char* data;
	int* next_idx;
	unsigned long long* data_as_uint64;
	int n;
	int l;
	int* indicies;
	unsigned long long num_solutions;
	
	int arr_l;
	std::vector<std::pair<int, int>> solution;
	std::unique_ptr<SolvingStrategy> s;
	void count_sort(int idx);
	inline unsigned char reverse_bits(unsigned char v) { return (v * 0x0202020202ULL & 0x010884422010ULL) % 1023; }
	friend class SolvingStrategyCPU_Naive;
	friend class SolvingStrategy_GPU;
	friend class SolvingStrategyCPU_Trie;
	StringsData(const char* filename, SolvingStrategy* s);
	StringsData(const StringsData& other) = delete;
	void solve() { s.get()->solve(*this); }
	void transpose_data();
	void reinterpret_as_uint64();
	void sort_data();
	void set_next_idx();
	bool cmp_sequences(int i, int j);
	void printSolution(const char* filename = NULL);
	~StringsData();

};

