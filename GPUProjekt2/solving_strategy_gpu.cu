#include "solving_strategy_gpu.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "cudaError.h"
#include "cuda_runtime.h"
#include "thrust/adjacent_difference.h"
#include "thrust/scan.h"
#include "thrust/equal.h"
#include "data.h"
#include "device_launch_parameters.h"
#include <algorithm>

// kernel odpowiedzialny za budowê drzewa
__global__ void createTreeKernel(unsigned int* N, int* S, unsigned char* data, int* begin, int n, int arr_l)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n)
	{
		// przypisanie dzieci do rodziców w drzewie
		for (int x = 0; x < arr_l - 1; x++)
		{
			N[(begin[x] + S[n * x + idx] - 1) * 256 + data[n * x + idx]] = S[n * (x + 1) + idx];
		}

		// przypisanie liœci w drzewie
		N[(begin[arr_l - 1] + S[n * (arr_l - 1) + idx] - 1) * 256 + data[n * (arr_l - 1) + idx]] = idx + 1;
	}
}

__global__ void searchTreeKernel(unsigned int* N, unsigned char* data, int* begin, int* result, int n, int arr_l, int l)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n)
	{
		// alokacja pamiêci na idx-ty ci¹g
		unsigned char* sequence = new unsigned char[arr_l];

		// przepisanie ci¹gu do pamiêci lokalnej
		for (int i = 0; i < arr_l; i++)
			sequence[i] = data[i * n + idx];

		for (int pos = 0; pos < l; pos++)
		{
			// modyfikacja ci¹gu na pos-tej pozycji
			sequence[pos >> 3] ^= (1 << (pos & 7));

			// przeszukiwanie drzewa i wypisanie odpowiedzi
			result[pos * n + idx] = search(N, sequence, begin, arr_l);

			// powrotna modyfikacja
			sequence[pos >> 3] ^= (1 << (pos & 7));
		}


		delete[] sequence;
	}
}

__device__ int search(unsigned int* N, unsigned char* value, int* begin, int arr_l)
{
	int pos = 1;
	for (int level = 0; level < arr_l; level++)
	{
		pos = N[((begin[level] + pos - 1) << 8) + value[level]];
		// dopóki nie natrafimy na wartoœæ 0 - idziemy po istniej¹cej œcie¿ce w drzewie
		if (pos == 0) break;
	}
	return pos - 1;
}


void SolvingStrategy_GPU::solve(SequencesData& data)
{
	// preprocessing
	data.sort_data();
	data.set_next_idx();
	data.transpose_data();
	
	// inicjalizacja drzewa
	HammingRTrie_GPU trie(data.data, data.n, data.arr_l, data.l);

	// przeszukiwanie drzewa
	int* res = trie.searchAll();


	// przeszukiwanie odpowiedzi - dla ka¿dego ci¹gu i dla ka¿dej pozycji sprawdzamy, czy nie otrzymaliœmy dopasowania
	for (int i = 0; i < data.l; i++)
	{
		for (int j = 0; j < data.n; j = data.next_idx[j])
		{
			if (res[i * data.n + j] >= 0 && res[i * data.n + j] > j)
			{
				// w tej linii upewniamy siê, ¿e wybieramy pierwszy indeks dla tego ci¹gu (je¿eli takich ci¹gów jest wiêcej w danych wejœciowych)
				int other = data.start_of_idx[res[i * data.n + j]];
				data.num_solutions += (unsigned long long)(data.next_idx[j] - j) * (data.next_idx[other] - other);
				if (verbose)
				{
					for (int n1 = j; n1 < data.next_idx[j]; n1++)
						for (int n2 = other; n2 < data.next_idx[other]; n2++)
							if (data.indicies[n1] < data.indicies[n2])
								data.solution.push_back(std::make_pair(data.indicies[n1], data.indicies[n2]));
							else data.solution.push_back(std::make_pair(data.indicies[n2], data.indicies[n1]));
				}
			}
		}
	}

	std::sort(data.solution.begin(), data.solution.end());
	delete[] res;
}

HammingRTrie_GPU::HammingRTrie_GPU(unsigned char* data, int n, int arr_l, int l)
{
	this->arr_l = arr_l;
	this->n = n;
	this->l = l;

	CALL(cudaSetDevice(0));
	V = thrust::device_vector<unsigned char>(data, data + n * arr_l);
	thrust::device_vector<int> S = thrust::device_vector<int>(n * arr_l, 0);


	// obliczamy F_0, ..., F_{arr_l - 1}
	S[0] = 1;
	for (int i = 1; i < arr_l; i++)
	{
		//thrust::transform(V.begin() + n * (i - 1), V.begin() + n * i - 1, V.begin() + n * (i - 1) + 1, S[i].begin() + 1, NotEqualsFunctor());
		thrust::adjacent_difference(
			V.begin() + n * (i - 1), 
			V.begin() + n * i,
			S.begin() + i * n);
		S[i * n] = 1;
		thrust::transform(
			S.begin() + (i - 1) * n, 
			S.begin() + i * n,
			S.begin() + i * n, 
			S.begin() + i * n,
			thrust::logical_or<int>());
	}
	
	// obliczamy S_0, ..., S_{arr_l - 1}
	for (int i = 0; i < arr_l; i++)
	{
		thrust::inclusive_scan(
			S.begin() + i * n, 
			S.begin() + (i + 1) * n, 
			S.begin() + i * n);
	}

	// obliczamy iloœæ wêz³ów w drzewie oraz indeks pocz¹tku ka¿dego z poziomów drzewa
	int* begin = new int[arr_l];
	total_nodes = 0;
	for (int i = 0; i < arr_l; i++)
	{
		begin[i] = total_nodes;
		total_nodes += S[(i + 1) * n - 1];
	}

	// inicjowanie pamiêci dla drzewa
	CALL(cudaMalloc((void**)&this->dev_begin, sizeof(int) * arr_l));
	CALL(cudaMemcpy(dev_begin, begin, sizeof(int) * arr_l, cudaMemcpyHostToDevice));

	CALL(cudaMalloc((void**)&this->N, total_nodes * 256 * sizeof(unsigned int)));
	CALL(cudaMemset(this->N, 0, total_nodes * 256 * sizeof(unsigned int)));

	delete[] begin;

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	// budowanie drzewa
	createTreeKernel<<<numBlocks, blockSize>>>(N, (int*)thrust::raw_pointer_cast(S.data()), (unsigned char*)thrust::raw_pointer_cast(V.data()), dev_begin, n, arr_l);
	
	CALL(cudaGetLastError());
	CALL(cudaDeviceSynchronize());

}
int* HammingRTrie_GPU::searchAll()
{
	// inicjowanie tablicy odpowiedzi
	int* result, *dev_result;
	result = new int[n * l];
	cudaMalloc((void**)&dev_result, sizeof(int) * n * l);
	cudaMemset(dev_result, 0, sizeof(int) * n * l);

	int blockSize = 64;
	int numBlocks = (n + blockSize - 1) / blockSize;

	// poszukiwanie w drzewie
	searchTreeKernel << <numBlocks, blockSize >> > (N, (unsigned char*)thrust::raw_pointer_cast(V.data()), dev_begin, dev_result, n, arr_l, l);
	CALL(cudaGetLastError());
	CALL(cudaDeviceSynchronize());

	// przekopiowanie odpowiedzi na CPU
	CALL(cudaMemcpy(result, dev_result, sizeof(int) * n * l, cudaMemcpyDeviceToHost));
	CALL(cudaFree(dev_result));

	return result;
}

HammingRTrie_GPU::~HammingRTrie_GPU()
{
	CALL(cudaFree(N));
	CALL(cudaFree(dev_begin));
}



