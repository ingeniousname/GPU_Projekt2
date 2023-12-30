#pragma once

class StringsData;

class SolvingStrategy
{
public:
	virtual void solve(StringsData& data) = 0;
};