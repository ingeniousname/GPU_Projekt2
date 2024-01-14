XCOPY .\..\x64\Debug\HammingOne.exe .
compute-sanitizer --tool memcheck ./HammingOne.exe ./data/test2.dat ./solution.txt 1 -v
PAUSE