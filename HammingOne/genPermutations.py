import sys, getopt

helpstring = """Skrypt służący do generowania <= n ciągów długości l, z czego każdy z nich zaczyna się ciągiem samych zer,
a kończy k-permutacją bitów, Dostępne opcje:
-n<liczba> - n = liczba
-l<liczba> - l = liczba
-k<liczba> - k = liczba
-o - ścieżka do plik wyjściowego
Przykładowe wywołanie:
python3 ./genPermutations.py -n200000 -k18 -l2000 -o ./data_200k.dat"""

def next_binary_num(s, l):
   if s[l - 1] == '1':
      i = l - 2
      while i > 0 and s[i] == '1':
         i = i - 1
      s[i] = '1'
      i = i + 1
      while i < l:
         s[i] = '0'
         i = i + 1
   else:
      s[l - 1] = '1'

l = 10000
k = 20
n = 100
out = "data.dat"
opts, args = getopt.getopt(sys.argv[1:],"hl:n:o:k:")
for opt, arg in opts:
   if opt == '-h':
      print (helpstring)
      sys.exit()
   elif opt == "-k":
      k = int(arg)
   elif opt == "-l":
      l = int(arg)
   elif opt == "-n":
      n = int(arg)
   elif opt == "-o":
      out = arg

base = str('')
perm = []
for i in range(0, l - k):
   base +='0'

for i in range(0, k):
   perm.append('0')
print(''.join(perm))
with open(out, "w+") as f:
   f.write(f"{n},{l}\n")
   for i in range(1, n-1):
      f.write(f"{base + str(''.join(map(str, perm)))}\n")
      next_binary_num(perm, k)
   f.write(f"{base + str(''.join(map(str, perm)))}")