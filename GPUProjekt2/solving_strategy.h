#pragma once

class StringsData;

class SolvingStrategy
{
protected:
	bool verbose;
public:
	SolvingStrategy(bool v) : verbose(v) {}
	virtual void solve(StringsData& data) = 0;
};