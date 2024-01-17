#include "solving_strategy_cpu_naive.h"
#include "data.h"
#include <algorithm>
#include <bit>

// obliczanie odleg�o�ci hamminga
bool SolvingStrategyCPU_Naive::hamming_dist_eq_to_one(SequencesData& data, int i, int j)
{
	int sol = 0;
	for (int k = 0; k < data.arr_l; k++)
	{
		sol += std::_Popcount<unsigned long long>(data.data_as_uint64[i * data.arr_l + k] ^ data.data_as_uint64[j * data.arr_l + k]);
	}
	return sol == 1;
}

void SolvingStrategyCPU_Naive::solve(SequencesData& data)
{
	// sortowanie nie jest tu potrzebne, jednak nie zwi�ksza ono z�o�ono�ci, a pozwala na pozbycie si� duplikat�w
	data.sort_data();
	data.set_next_idx();
	data.reinterpret_as_uint64();

	// por�wnujemy ze sob� wszystkie pary unikatowych ci�g�w
	for (int i = 0; i < data.n; i = data.next_idx[i])
	{
		for(int j = data.next_idx[i]; j < data.n; j = data.next_idx[j])
			if (this->hamming_dist_eq_to_one(data, i, j))
			{
				data.num_solutions += (unsigned long long)(data.next_idx[i] - i) * (data.next_idx[j] - j);
				if (verbose)
				{
					for (int n1 = i; n1 < data.next_idx[i]; n1++)
						for (int n2 = j; n2 < data.next_idx[j]; n2++)
							// dla zachowania przejrzysto�ci w odpowiedzi
							if (data.indicies[n1] < data.indicies[n2])
								data.solution.push_back(std::make_pair(data.indicies[n1], data.indicies[n2]));
							else data.solution.push_back(std::make_pair(data.indicies[n2], data.indicies[n1]));
				}
			}
	}

	// dla przejrzysto�ci
	std::sort(data.solution.begin(), data.solution.end());
}