#pragma once
#include "cuda_runtime.h"
#include <cstdio>
#include <cstdlib>

// Makro pozwalaj�ce na sprawdzenie b��du podczas wywo�ywania funkcji CUDA
#define CALL(x) do{ \
	cudaError_t cudaStatus = x; \
	if(cudaStatus != cudaSuccess) \
	{ \
		fprintf(stderr, "CudaError: %s, %s\n", cudaGetErrorName(cudaStatus), cudaGetErrorString(cudaStatus)); \
		exit(1); \
	} \
} while(false) // zastosowanie konstrukcji do-while jest celowe - mo�na postawi� �rednik na ko�cu gdy owijamy funkcj� w makro