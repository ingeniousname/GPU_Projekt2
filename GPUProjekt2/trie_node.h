#pragma once
#include <cstring>

class TrieNode
{
public:
	int children[256];
	TrieNode()
	{
		memset(children, 0, 256 * sizeof(int));
	}

};
