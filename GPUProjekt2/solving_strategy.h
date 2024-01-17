#pragma once

class SequencesData;

// Klasa bazowa dla algorytm�w rozwi�zuj�cych problem HammingOne
class SolvingStrategy
{
protected:
	bool verbose;
public:
	SolvingStrategy(bool v) : verbose(v) {}
	virtual void solve(SequencesData& data) = 0;
};