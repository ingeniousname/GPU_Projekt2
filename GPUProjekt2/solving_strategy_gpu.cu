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

__global__ void createTreeKernel(int* N, int* S, unsigned char* data, int* begin, int n, int arr_l)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	begin[0] = 0;
	if (idx < n)
	{
		for (int x = 0; x < arr_l - 1; x++)
		{
			N[(begin[x] + S[n * x + idx] - 1) * 256 + data[n * x + idx]] = S[n * (x + 1) + idx];
			begin[x + 1] = S[n * (x + 1) - 1];
		}
		N[(begin[arr_l - 1] + S[n * (arr_l - 1) + idx] - 1) * 256 + data[n * (arr_l - 1) + idx]] = idx + 1;
	}
}

__global__ void searchTree(int* tree, unsigned char* data, int* begin, int* result, int n, int arr_l, int l)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n)
	{
		unsigned char* sequence = new unsigned char[arr_l];
		for (int i = 0; i < arr_l; i++)
			sequence[i] = data[i * n + idx];

		for (int pos = 0; pos < l; pos++)
		{
			sequence[pos / 8] ^= (1 << (pos % 8));
			result[pos * n + idx] = search(tree, sequence, begin, arr_l);
			sequence[pos / 8] ^= (1 << (pos % 8));
		}


		delete[] sequence;

	}
}

__device__ int search(int* tree, unsigned char* value, int* begin, int arr_l)
{
	int pos = 1;
	for (int level = 0; level < arr_l; level++)
	{
		pos = tree[(begin[level] + pos - 1) * 256 + value[level]];
		if (pos == 0) return -1;
	}
	return pos - 1;
}


void SolvingStrategy_GPU::solve(StringsData& data)
{
	data.sort_data();
	data.set_next_idx();
	data.transpose_data();
	HammingRTrie_GPU trie(data.data, data.n, data.arr_l, data.l);
	int* res = trie.searchAll();

	for (int i = 0; i < data.l; i++)
	{
		for (int j = 0; j < data.n; j = data.next_idx[j])
		{
			if (res[i * data.n + j] != 0 && res[i * data.n + j] > j)
			{
				int other = res[i * data.n + j];
				data.num_solutions += (data.next_idx[j] - j) * (data.next_idx[other] - other);
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
	thrust::device_vector<int>S = thrust::device_vector<int>(n * arr_l, 0);

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
	
	for (int i = 0; i < arr_l; i++)
	{
		thrust::inclusive_scan(
			S.begin() + i * n, 
			S.begin() + (i + 1) * n, 
			S.begin() + i * n);
	}

	total_nodes = 0;
	for (int i = 0; i < arr_l; i++)
		total_nodes += S[(i + 1) * n - 1];
	/*for (int i = 0; i < arr_l; i++)
	{
		for (int j = 0; j < n; j++)
			std::cout << S[i * n + j] << " ";
		std::cout << "\n";
	}*/
	CALL(cudaMalloc((void**)&this->N, total_nodes * 256 * sizeof(int)));
	CALL(cudaMemset(this->N, 0, total_nodes * 256 * sizeof(int)));
	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;
	CALL(cudaMalloc((void**)&this->dev_begin, sizeof(int) * arr_l));
	createTreeKernel<<<numBlocks, blockSize>>>(N, (int*)thrust::raw_pointer_cast(S.data()), (unsigned char*)thrust::raw_pointer_cast(V.data()), dev_begin, n, arr_l);
	
	CALL(cudaGetLastError());
	CALL(cudaDeviceSynchronize());

	//N = new TrieNode*[arr_l];
	//for (int x = 0; x < arr_l - 1; x++)
	//{
	//	N[x] = new TrieNode[S[(x + 1) * n - 1]];
	//	for (int i = 0; i < n; i++)
	//	{
	//		N[x][S[n * x + i] - 1].children[data[n * x + i]] = S[n * (x + 1) + i];
	//	}
	//}
	//int count = 0;
	//int x = arr_l - 1;
	//N[x] = new TrieNode[S[(x + 1) * n - 1]];
	//for (int i = 0; i < n; i++)
	//{
	//	if (N[x][S[n * x + i] - 1].children[data[n * x + i]] == 0)
	//		N[x][S[n * x + i] - 1].children[data[n * x + i]] = ++count;
	//}
	//child_ref = std::vector<std::vector<int>>(count, std::vector<int>());

	//for (int i = 0; i < n; i++)
	//	child_ref[N[x][S[n * x + i] - 1].children[data[n * x + i]] - 1].push_back(i);


}
int* HammingRTrie_GPU::searchAll()
{
	int* result, *dev_result;
	result = new int[n * l];
	cudaMalloc((void**)&dev_result, sizeof(int) * n * l);
	cudaMemset(dev_result, 0, sizeof(int) * n * l);

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	searchTree << <numBlocks, blockSize >> > (N, (unsigned char*)thrust::raw_pointer_cast(V.data()), dev_begin, dev_result, n, arr_l, l);
	CALL(cudaGetLastError());
	CALL(cudaDeviceSynchronize());

	CALL(cudaMemcpy(result, dev_result, sizeof(int) * n * l, cudaMemcpyDeviceToHost));
	CALL(cudaFree(dev_result));

	return result;
}

HammingRTrie_GPU::~HammingRTrie_GPU()
{
	CALL(cudaFree(N));
	CALL(cudaFree(dev_begin));
}



