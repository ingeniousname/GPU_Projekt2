#include "solving_strategy_cpu_trie.h"
#include "data.h"
#include <cstring>
#include <algorithm>

HammingRTrie::HammingRTrie(unsigned char* data, int n, int arr_l)
{
	this->arr_l = arr_l;
	this->n = n;
	int* S = new int[n * arr_l];
	memset(S, 0, n * arr_l * sizeof(int));
	for (int i = 0; i < n; i++)
		S[i] = 1;
	for (int x = 1; x < arr_l; x++)
	{
		S[x * n] = 1;
		for (int i = 1; i < n; i++)
		{
			S[x * n + i] = S[x * n + i - 1] + (data[(x - 1) * n + i] != data[(x - 1) * n + i - 1]);
		}

	}

	N = new TrieNode*[arr_l];
	for (int x = 0; x < arr_l; x++)
	{
		N[x] = new TrieNode[S[(x + 1) * n - 1]];
		for (int i = 0; i < n; i++)
		{
			if(x == arr_l - 1)
				N[x][S[n * x + i] - 1].children[data[n * x + i]] = i + 1;
			else N[x][S[n * x + i] - 1].children[data[n * x + i]] = S[n * (x + 1) + i];
			
		}
	}



}

int HammingRTrie::search(unsigned char* value)
{
	int pos = 1;
	for (int i = 0; i < arr_l; i++)
	{
		pos = N[i][pos - 1].children[value[i]];
		if (pos == 0) return -1;
	}
	return pos - 1;
}

HammingRTrie::~HammingRTrie()
{
	for (int i = 0; i < arr_l; i++)
		delete[] N[i];
	delete[] N;
}

bool SolvingStrategyCPU_Trie::hamming_dist_eq_to_one(StringsData& data, int i, int j)
{
	return false;
}

void SolvingStrategyCPU_Trie::solve(StringsData& data)
{
	data.sort_data();
	data.transpose_data();

	HammingRTrie trie(data.data, data.n, data.arr_l);
	unsigned char* v = new unsigned char[data.arr_l];

	for (int i = 0; i < data.n; i++)
	{
		for (int j = 0; j < data.arr_l; j++)
			v[j] = data.data[j * data.n + i];
		
		for (int pos = 0; pos < data.l; pos++)
		{
			v[pos / 8] ^= (1 << (pos % 8));
			int res = trie.search(v);
			if (res >= 0)
			{
				if(data.indicies[i] < data.indicies[res])
					data.solution.push_back(std::make_pair(data.indicies[i], data.indicies[res]));
			}
			v[pos / 8] ^= (1 << (pos % 8));

		}
		
	}
	std::sort(data.solution.begin(), data.solution.end());
	delete[] v;
}
