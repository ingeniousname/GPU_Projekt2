#include <iostream>
#include "data.h"
#include "solving_strategy_cpu_naive.h"
#include "solving_strategy_cpu_trie.h"
#include "solving_strategy_gpu.cuh"
#include "Timer.h"


int main(int argc, char* argv[])
{
	if (argc < 4 || argc > 6 || (argc == 5 && strcmp(argv[4], "-v")))
	{
		printf("UZYCIE: ./HammingOne <plik_wejsciowy> <plik_wyjsciowy> <numer trybu> <-v>\nTryby dzialania:\n0 - CPU (algorytm naiwny)\n1 - CPU (algorytm Trie)\n2 - GPU (algorytm Trie)\nParametr -v jest opcjonalny - oznacza on wypisanie par indeksow ciagow, dla ktorych odleglosc Hamminga jest rowna 1. Prosze ten parametr zawsze umieszczac na koncu.\n");
		return 1;
	}

	printf("%s\n", argv[1]);
	const int mode = atoi(argv[3]);
	bool verbose = (argc == 5);
	SolvingStrategy* s;
	switch (mode)
	{
		case 0: s = new SolvingStrategyCPU_Naive(verbose); break;
		case 1: s = new SolvingStrategyCPU_Trie(verbose); break;
		case 2: s = new SolvingStrategy_GPU(verbose); break;
		default:
		{
			printf("Nieznany tryb!\n");
			return 1;
		}
	}





	StringsData data(argv[1], s);
	{
		Timer_CPU t("Solving", true);
		data.solve();
	
	}
	{
	
		Timer_CPU t("Output printing", true);
		data.printSolution(argv[2]);
	}
}
