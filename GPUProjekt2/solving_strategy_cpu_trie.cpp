#include "solving_strategy_cpu_trie.h"
#include "data.h"
#include <cstring>
#include <algorithm>
#include <fstream>
#include <set>
#include <iostream>

HammingRTrie_CPU::HammingRTrie_CPU(unsigned char* data, int n, int arr_l)
{
	this->arr_l = arr_l;
	this->n = n;
	// tworzymy pojedyncza tablicê, która na pocz¹tku funkcjonuje jako tablica F, a nastêpnie jako S
	int* S = new int[n * arr_l];
	memset(S, 0, n * arr_l * sizeof(int));
	S[0] = 1;
	// obliczamy F_1, F_2, ..., F_{arr_l - 1}
	for (int x = 1; x < arr_l; x++)
	{
		S[x * n] = 1;
		for (int i = 1; i < n; i++)
		{
			S[x * n + i] = S[(x - 1) * n + i] | (data[(x - 1) * n + i] != data[(x - 1) * n + i - 1]);
		}

	}

	// obliczamy sumy prefiksowe (S_0, ..., S_{arr_l - 1})
	for (int x = 0; x < arr_l; x++)
	{
		for (int i = 1; i < n; i++)
		{
			S[x * n + i] += S[x * n + i - 1];
		}

	}

	// inicjujemy drzewo
	N = new TrieNode*[arr_l];
	for (int x = 0; x < arr_l - 1; x++)
	{
		N[x] = new TrieNode[S[(x + 1) * n - 1]];
		// przypisanie dzieci w drzewie
		for (int i = 0; i < n; i++)
		{
			N[x][S[n * x + i] - 1].children[data[n * x + i]] = S[n * (x + 1) + i];
		}
	}

	// przypisanie liœci w drzewie, indeksy ci¹gów w liœciach
	int x = arr_l - 1;
	N[x] = new TrieNode[S[(x + 1) * n - 1]];
	for (int i = 0; i < n; i++)
	{
		if (N[x][S[n * x + i] - 1].children[data[n * x + i]] == 0)
			N[x][S[n * x + i] - 1].children[data[n * x + i]] = i + 1;
	}



}

int HammingRTrie_CPU::search(unsigned char* value)
{
	int pos = 1;
	for (int level = 0; level < arr_l; level++)
	{
		pos = N[level][pos - 1].children[value[level]];
		// dopóki nie natrafimy na wartoœæ 0 - idziemy po istniej¹cej œcie¿ce w drzewie
		if (pos == 0) break;
	}
	return pos - 1;
}

HammingRTrie_CPU::~HammingRTrie_CPU()
{
	for (int i = 0; i < arr_l; i++)
		delete[] N[i];
	delete[] N;
}

void SolvingStrategyCPU_Trie::solve(SequencesData& data)
{
	// preprocessing
	data.sort_data();
	data.set_next_idx();
	data.transpose_data();

	// budowa drzewa
	HammingRTrie_CPU trie(data.data, data.n, data.arr_l);
	unsigned char* v = new unsigned char[data.arr_l];

	for(int i = 0; i < data.n; i = data.next_idx[i])
	{
		// przepisanie ci¹gu do lokalnej tablicy
		for (int j = 0; j < data.arr_l; j++)
			v[j] = data.data[j * data.n + i];

		for (int pos = 0; pos < data.l; pos++)
		{
			// modyfikacja ci¹gu na l-tym bicie
			v[pos << 3] ^= (1 << (pos & 7));

			// przeszukiwanie drzewa
			int res = trie.search(v);

			// znalezienie odpowiedzi, upewniamy siê, ¿e siê nie powtarzamy
			if (res >= 0 && i < res)
			{
				data.num_solutions += (unsigned long long)(data.next_idx[i] - i) * (data.next_idx[res] - res);
				if (verbose)
				{
					for (int n1 = i; n1 < data.next_idx[i]; n1++)
						for (int n2 = res; n2 < data.next_idx[res]; n2++)
							if (data.indicies[n1] < data.indicies[n2])
								data.solution.push_back(std::make_pair(data.indicies[n1], data.indicies[n2]));
							else data.solution.push_back(std::make_pair(data.indicies[n2], data.indicies[n1]));
				}
					
			}
			// modyfikacja powrotna
			v[pos << 3] ^= (1 << (pos & 7));
		}

	}

	// dla przejrzystoœci odpowiedzi
	std::sort(data.solution.begin(), data.solution.end());
	delete[] v;
}
