#pragma once
#include <vector>
#include "solving_strategy.h"
#include <memory>

class StringsData
{
	unsigned long long* data;
	int n;
	int l;
	int arr_l;
	std::vector<std::pair<int, int>> solution;
	std::unique_ptr<SolvingStrategy> s;
public:
	friend class SolvingStrategyCPU;
	StringsData(const char* filename, SolvingStrategy* s);
	StringsData(const StringsData& other) = delete;
	void solve() { s.get()->solve(*this); }
	void printSolution(const char* filename = NULL);
	~StringsData();

};

