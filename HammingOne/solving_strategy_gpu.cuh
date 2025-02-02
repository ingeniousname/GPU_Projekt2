#pragma once
#include "solving_strategy.h"
#include "trie_node.h"
#include "cuda_runtime.h"

#include <thrust\device_vector.h>

__global__ void createTreeKernel(unsigned int* tree, int* S, unsigned char* data, int* dev_begin, int n, int arr_l);
__global__ void searchTreeKernel(unsigned int* tree, unsigned char* data, int* dev_begin, int* result, int n, int arr_l, int l);
__device__ int search(unsigned int* tree, unsigned char* value, int* dev_begin, int arr_l);


struct NotEqualsFunctor {
	template <typename T>
	__host__ __device__
		int operator()(const T& t1, const T& t2) {
		return t1 != t2;
	}
};

class HammingRTrie_GPU
{
	int arr_l, n, l, total_nodes;
	int* dev_begin;
	thrust::device_vector<unsigned char> V;
public:
	unsigned int* N;
	HammingRTrie_GPU(unsigned char* data, int n, int arr_l, int l);
	int* searchAll();
	~HammingRTrie_GPU();
};


class SolvingStrategy_GPU : public SolvingStrategy
{
public:
	SolvingStrategy_GPU(bool v) : SolvingStrategy(v) {}
	virtual void solve(SequencesData& data);
};
			
