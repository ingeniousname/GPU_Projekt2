#include "data.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include "Timer.h"

// implementacja sortowania przez zliczanie
void SequencesData::count_sort(int idx)
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


SequencesData::SequencesData(const char* filename, SolvingStrategy* s) : s(s), transposed(false)
{
	Timer_CPU t("Data reading", true);
	size_t typesize = sizeof(unsigned char) * 8;

	// odczytywanie ci¹gów z pliku
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
	else
	{
		std::cout << "Failed to open file: " << filename << "\n";
		exit(1);
	}
	f.close();

	// ustanowienie indeksów ci¹gów, bêd¹ siê one zmienia³y podczas sortowania
	indicies = new int[n];
	for(int i = 0; i < n; i++)
		indicies[i] = i;

	this->num_solutions = 0;
}

// funkcja transponuj¹ca dane (zamiana z AoS na SoA)
void SequencesData::transpose_data()
{
	transposed = !transposed;
	unsigned char* transposed_data = new unsigned char[n * arr_l];
	for (int i = 0; i < arr_l; i++)
		for (int j = 0; j < n; j++)
			transposed_data[i * n + j] = data[j * arr_l + i];
	delete[] data;
	data = transposed_data;

}

// funkcja sortuj¹ca dane
void SequencesData::sort_data()
{
	for (int i = arr_l - 1; i >= 0; i--)
		count_sort(i);
}

// funkcja wypisuj¹ca dane do pliku
void SequencesData::printSolution(const char* filename)
{
	if (filename == NULL)
	{
		for (auto& sol : this->solution)
			std::cout << "(" << sol.first << ", " << sol.second << ")\n";
	}
	else
	{
		std::ofstream f(filename, std::ios::out);
		f << num_solutions << "\n";
		for (auto& sol : this->solution)
			f << "(" << sol.first << ", " << sol.second << ")\n";
		f.close();
	}
}

// funkcja, która wywo³ana po sortowaniu odpowiednio przypisuje indeksy do tablic next_idx oraz start_of_idx tak, ¿e duplikaty ci¹gów s¹ traktowane równowa¿nie
void SequencesData::set_next_idx()
{
	next_idx = new int[this->n];
	start_of_idx = new int[this->n];
	int i = 0, j = 1;
	while (j < this->n)
	{
		if (cmp_sequences(i, j))
			j++;
		else
		{
			next_idx[i] = j;
			int start = i;
			for (i; i < j; i++)
				start_of_idx[i] = start;
			i = j;
			j = i + 1;
		}
	}
	next_idx[i] = j;
	int start = i;
	for (i; i < j; i++)
		start_of_idx[i] = start;
}

// funkcja porónuj¹ca dwa ci¹gi
bool SequencesData::cmp_sequences(int i, int j)
{
	for (int k = 0; k < arr_l; k++)
	{
		if (transposed)
		{
			if (data[k * n + i] != data[k * n + j])
				return false;
		}
		else
		{
			if (data[i * arr_l + k] != data[j * arr_l + k])
				return false;
		}

	}
	return true;
}

SequencesData::~SequencesData()
{
	delete[] data;
	delete[] indicies;
	delete[] next_idx;
	delete[] start_of_idx;
}

// funkcja pozwalaj¹ca przepisaæ dane do tablicy typu unsigned long long - przydatne w przypadku algorytmu naiwnego
void SequencesData::reinterpret_as_uint64()
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
