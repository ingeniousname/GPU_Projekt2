#include "solving_strategy_cpu_trie.h"
#include "data.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <set>

HammingRTrie::HammingRTrie(unsigned char* data, int n, int arr_l)
{
	this->arr_l = arr_l;
	this->n = n;
	int* S = new int[n * arr_l];
	memset(S, 0, n * arr_l * sizeof(int));
	S[0] = 1;
	for (int x = 1; x < arr_l; x++)
	{
		S[x * n] = 1;
		for (int i = 1; i < n; i++)
		{
			S[x * n + i] = S[(x - 1) * n + i] | (data[(x - 1) * n + i] != data[(x - 1) * n + i - 1]);
		}

	}

	for (int x = 0; x < arr_l; x++)
	{
		for (int i = 1; i < n; i++)
		{
			S[x * n + i] += S[x * n + i - 1];
		}

	}

	N = new TrieNode*[arr_l];
	for (int x = 0; x < arr_l - 1; x++)
	{
		N[x] = new TrieNode[S[(x + 1) * n - 1]];
		for (int i = 0; i < n; i++)
		{
			N[x][S[n * x + i] - 1].children[data[n * x + i]] = S[n * (x + 1) + i];
		}
	}
	int count = 0;
	int x = arr_l - 1;
	N[x] = new TrieNode[S[(x + 1) * n - 1]];
	for (int i = 0; i < n; i++)
	{
		if (N[x][S[n * x + i] - 1].children[data[n * x + i]] == 0)
			N[x][S[n * x + i] - 1].children[data[n * x + i]] = ++count;
	}
	child_ref = std::vector<std::vector<int>>(count, std::vector<int>());

	for (int i = 0; i < n; i++)
		child_ref[N[x][S[n * x + i] - 1].children[data[n * x + i]] - 1].push_back(i);


}

int HammingRTrie::search(unsigned char* value)
{
	int pos = 1;
	for (int level = 0; level < arr_l; level++)
	{
		pos = N[level][pos - 1].children[value[level]];
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
	std::set<int> inputs;
	for (int i = 0; i < data.n; i++)
		inputs.insert(i);

	data.sort_data();
	data.transpose_data();

	HammingRTrie trie(data.data, data.n, data.arr_l);
	unsigned char* v = new unsigned char[data.arr_l];

	while(!inputs.empty())
	{
		int i = *inputs.begin();
		for (int j = 0; j < data.arr_l; j++)
			v[j] = data.data[j * data.n + i];

		int idx1 = trie.search(v);

		for (int pos = 0; pos < data.l; pos++)
		{
			v[pos / 8] ^= (1 << (pos % 8));
			int res = trie.search(v);
			if (res >= 0)
			{
				if (idx1 < res)
				{
					for(int n1 : trie.child_ref[idx1])
						for (int n2 : trie.child_ref[res])
						{
							if (data.indicies[n1] < data.indicies[n2])
								data.solution.push_back(std::make_pair(data.indicies[n1], data.indicies[n2]));
							else data.solution.push_back(std::make_pair(data.indicies[n2], data.indicies[n1]));
						}
				}
			}
			v[pos / 8] ^= (1 << (pos % 8));

		}
		for (int n1 : trie.child_ref[idx1])
			inputs.erase(n1);

	}

	std::sort(data.solution.begin(), data.solution.end());
	delete[] v;
}
