#include "data.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include "Timer.h"

StringsData::StringsData(const char* filename, SolvingStrategy* s) : s(s)
{
	Timer_CPU t("Data reading", true);
	size_t typesize = sizeof(unsigned long long) * 8;

	std::ifstream f(filename, std::ios::in);
	if (f.is_open())
	{
		f >> this->n;
		f.get();
		f >> this->l;

		char* temp_data_string = new char[this->l + 1];
		this->arr_l = (int)std::ceil((double)this->l / typesize);
		this->data = new unsigned long long[this->n * this->arr_l];
		memset(this->data, 0, (this->n * this->arr_l) * sizeof(unsigned long long));



		while (f.get() != '\n');
		long long bitoffset = 0, typeoffset = 0;
		for(int k = 0; k < this->n; k++)
		{
			f.getline(temp_data_string, this->l + 1);
			bitoffset = 0;
			typeoffset = arr_l * k;
			for (int i = 0; i < this->l; i++)
			{
				data[typeoffset] |= (long long)(temp_data_string[i] - '0') << bitoffset;
				bitoffset++;
				if (bitoffset == typesize)
				{
					bitoffset = 0;
					typeoffset++;
				}
			}
		}
		delete[] temp_data_string;
	}
	f.close();
}

void StringsData::printSolution(const char* filename)
{
	if (filename == NULL)
	{
		for (auto& sol : this->solution)
			std::cout << "(" << sol.first << ", " << sol.second << ")\n";
	}
	else
	{
		std::ofstream f(filename, std::ios::out);
		for (auto& sol : this->solution)
			f << "(" << sol.first << ", " << sol.second << ")\n";
		f.close();
	}
}

StringsData::~StringsData()
{
	delete[] data;
}
