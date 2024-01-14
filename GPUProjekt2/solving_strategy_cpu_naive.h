#pragma once
#include "solving_strategy.h"

class SolvingStrategyCPU_Naive : public SolvingStrategy
{
	bool hamming_dist_eq_to_one(StringsData& data, int i, int j);
public:
	SolvingStrategyCPU_Naive(bool v) : SolvingStrategy(v) {}
	virtual void solve(StringsData& data);

};