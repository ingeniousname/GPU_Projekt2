XCOPY .\..\x64\Release\HammingOne.exe .
compute-sanitizer --tool memcheck ./HammingOne.exe ./data/data_200k.dat ./solution.txt 2 -v
PAUSE