#pragma once
#include "solving_strategy.h"
#include <cstring>
#include <vector>


class TrieNode
{
public:
	int children[256];
	TrieNode()
	{
		memset(children, 0, 256 * sizeof(int));
	}

};

class HammingRTrie
{
	int arr_l, n;
public:
	TrieNode** N;
	std::vector<std::vector<int>> child_ref;
	HammingRTrie(unsigned char* data, int n, int arr_l);
	int search(unsigned char* value);
	~HammingRTrie();
};

class SolvingStrategyCPU_Trie : public SolvingStrategy
{
	bool hamming_dist_eq_to_one(StringsData& data, int i, int j);
public:
	virtual void solve(StringsData& data);
};