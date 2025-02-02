#pragma once
#include "solving_strategy.h"
#include "trie_node.h"
#include <cstring>
#include <vector>



class HammingRTrie_CPU
{
	int arr_l, n;
public:
	TrieNode** N;
	HammingRTrie_CPU(unsigned char* data, int n, int arr_l);
	int search(unsigned char* value);
	~HammingRTrie_CPU();
};

class SolvingStrategyCPU_Trie : public SolvingStrategy
{
public:
	SolvingStrategyCPU_Trie(bool v) : SolvingStrategy(v) {}
	virtual void solve(SequencesData& data);
};