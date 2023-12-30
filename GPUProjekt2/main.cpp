#include <iostream>
#include "data.h"
#include "solving_strategy_cpu.h"
#include "Timer.h"



int main()
{
	StringsData data("./data/test2.dat", new SolvingStrategyCPU());
	{
		Timer_CPU t("Solving", true);
		data.solve();

	}
	{

		Timer_CPU t("Output printing", true);
		data.printSolution("solution.txt");
	}
}