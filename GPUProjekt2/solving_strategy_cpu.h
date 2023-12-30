#pragma once
#include "solving_strategy.h"

class SolvingStrategyCPU : public SolvingStrategy
{
	bool hamming_dist_eq_to_one(StringsData& data, int i, int j);
public:
	virtual void solve(StringsData& data);

};