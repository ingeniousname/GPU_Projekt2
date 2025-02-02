import random
import sys, getopt

def rand_sequence(p):
    key1 = ""
    for i in range(p):
        temp = str(random.randint(0, 1))
        key1 += temp
    return key1

helpstring = """Skrypt służący do generowania n losowych ciągów binarnych długości l
-n<liczba> - n = liczba
-l<liczba> - l = liczba
-k<liczba> - k = liczba
-o - ścieżka do plik wyjściowego
Przykładowe wywołanie:
python3 ./genRandom.py -n100000 -l1000 -o ./data_100k.dat"""

l = 10000
n = 100


out = "data.dat"
opts, args = getopt.getopt(sys.argv[1:],"hl:n:o:")
for opt, arg in opts:
   if opt == '-h':
      print (helpstring)
      sys.exit()
   elif opt == "-l":
      l = int(arg)
   elif opt == "-n":
      n = int(arg)
   elif opt == "-o":
      out = arg

with open(out, "w+") as f:
   f.write(f"{n},{l}\n")
   for i in range(1, n-1):
      f.write(f"{rand_sequence(l)}\n")
   f.write(f"{rand_sequence(l)}")