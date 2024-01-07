#include "solving_strategy_cpu_naive.h"
#include "data.h"
#include <bit>

bool SolvingStrategyCPU_Naive::hamming_dist_eq_to_one(StringsData& data, int i, int j)
{
	int sol = 0;
	for (int k = 0; k < data.arr_l; k++)
	{
		sol += std::_Popcount<unsigned long long>(data.data_as_uint64[i * data.arr_l + k] ^ data.data_as_uint64[j * data.arr_l + k]);
	}
	return sol == 1;
}

void SolvingStrategyCPU_Naive::solve(StringsData& data)
{

	data.reinterpret_as_uint64();
	for (int i = 0; i < data.n; i++)
	{
		for(int j = i + 1; j < data.n; j++)
			if(this->hamming_dist_eq_to_one(data, i, j))
				data.solution.push_back(std::make_pair(i, j));
	}

}