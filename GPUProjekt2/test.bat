XCOPY .\..\x64\Debug\HammingOne.exe .
compute-sanitizer --tool memcheck ./HammingOne.exe ./data/test3.dat ./solution.txt 2 -v
PAUSE