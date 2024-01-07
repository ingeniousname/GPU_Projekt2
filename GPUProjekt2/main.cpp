#include <iostream>
#include "data.h"
#include "solving_strategy_cpu_naive.h"
#include "solving_strategy_cpu_trie.h"
#include "Timer.h"



int main()
{
	StringsData data("./data/test3.dat", new SolvingStrategyCPU_Naive());
	/*data.sort_data();
	data.transpose_data();
	data.print_data();*/
	//HammingRTrie trie(data.data, data.n, data.arr_l);
	//unsigned char v[2] = { 245, 87 };
	{
		Timer_CPU t("Solving", true);
		data.solve();
	
	}
	{
	
		Timer_CPU t("Output printing", true);
		data.printSolution("solution.txt");
	}
}