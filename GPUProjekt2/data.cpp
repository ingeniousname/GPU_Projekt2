#include "data.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include "Timer.h"

void StringsData::count_sort(int idx)
{
	unsigned char* output = new unsigned char[n * arr_l];
	int* new_indicies = new int[n];
	int count[256];
	memset(count, 0, 256 * sizeof(int));
	for (int i = 0; i < n; i++)
		count[data[arr_l * i + idx]]++;

	for (int i = 1; i < 256; i++)
		count[i] += count[i - 1];

	for (int i = n - 1; i >= 0; i--)
	{
		memcpy(output + (count[data[i * arr_l + idx]] - 1) * arr_l, data + i * arr_l, arr_l);
		new_indicies[count[data[i * arr_l + idx]] - 1] = indicies[i];
		count[data[i * arr_l + idx]]--;
	}

	memcpy(data, output, arr_l * n);
	memcpy(indicies, new_indicies, n * sizeof(int));

	delete[] output;
	delete[] new_indicies;
}

StringsData::StringsData(const char* filename, SolvingStrategy* s) : s(s), transposed(false)
{
	Timer_CPU t("Data reading", true);
	size_t typesize = sizeof(unsigned char) * 8;

	std::ifstream f(filename, std::ios::in);
	if (f.is_open())
	{
		f >> this->n;
		f.get();
		f >> this->l;

		char* temp_data_string = new char[this->l + 1];
		this->arr_l = (int)std::ceil((double)this->l / typesize);
		this->data = new unsigned char[this->n * this->arr_l];
		memset(this->data, 0, (this->n * this->arr_l) * sizeof(unsigned char));



		while (f.get() != '\n');
		long long bitoffset = 0, typeoffset = 0;
		for(int k = 0; k < this->n; k++)
		{
			f.getline(temp_data_string, this->l + 1);
			bitoffset = 0;
			typeoffset = arr_l * k;
			for (int i = 0; i < this->l; i++)
			{
				data[typeoffset] |= (unsigned char)(temp_data_string[i] - '0') << bitoffset;
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

	//niepotrzebne
	/*for (int i = 0; i < n * arr_l; i++)
	{
	 	data[i] = reverse_bits(data[i]);
	}*/
	indicies = new int[n];
	for(int i = 0; i < n; i++)
		indicies[i] = i;
}

void StringsData::transpose_data()
{
	transposed = !transposed;
	unsigned char* transposed_data = new unsigned char[n * arr_l];
	for (int i = 0; i < arr_l; i++)
		for (int j = 0; j < n; j++)
			transposed_data[i * n + j] = data[j * arr_l + i];
	delete[] data;
	data = transposed_data;

}

void StringsData::sort_data()
{
	for (int i = arr_l - 1; i >= 0; i--)
		count_sort(i);
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

void StringsData::print_data()
{
	if (transposed)
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < arr_l; j++)
			{
				int mask = 0x80;
				for (int bit = 7; bit >= 0; bit--)
				{
					printf("%d", (data[i + j * n] & mask) >> bit);
					mask >>= 1;
				}
				printf("\'");
			}
			printf(" - %d\n", indicies[i]);
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < arr_l; j++)
			{
				int mask = 0x80;
				for (int bit = 7; bit >= 0; bit--)
				{
					printf("%d", (data[indicies[i] * arr_l + j] & mask) >> bit);
					mask >>= 1;
				}
				printf("\'");
			}
			printf("\n");
		}
	}


}

StringsData::~StringsData()
{
	delete[] data;
	delete[] indicies;
}

void StringsData::reinterpret_as_uint64()
{

	int new_arrl = (int)(std::ceil((double)this->l / 64));
	data_as_uint64 = new unsigned long long[n * new_arrl];
	memset(data_as_uint64, 0, n * new_arrl * sizeof(unsigned long long));
	for (int i = 0; i < n; i++)
	{
		memcpy(data_as_uint64 + i * new_arrl, data + i * arr_l, arr_l);

	}
	arr_l = new_arrl; 
	
}
