#pragma once

class SequencesData;

// Klasa bazowa dla algorytmów rozwi¹zuj¹cych problem HammingOne
class SolvingStrategy
{
protected:
	bool verbose;
public:
	SolvingStrategy(bool v) : verbose(v) {}
	virtual void solve(SequencesData& data) = 0;
};