<h2>GPU-accelerated Hamming One</h2>

Hamming distance is a number of positions where two strings differ. It can also be interpreted as a minimum number of substitiutions one has to make to transform one of the strings to another. For example binary strings:

<span style="color: green;">100</span><span style="color: red;">10</span><span style="color: green;">1</span>  
<span style="color: green;">100</span><span style="color: red;">01</span><span style="color: green;">1</span>  

have a hamming distance equal to 2.

For a input consisted of $n$ binary strings of length $l$, program finds how many pairs of strings have the hamming one distance equal to one and optionally writes those pairs out.

The program takes an input file with parameters $n$ and $l$ in the first line, and binary strings of length $l$ in the subsequent $n$ lines.

There are three calculation strategies to choose from:
- Naive CPU implementation (pairwise calculation)
- Trie implementation on CPU
- Trie implementation on GPU

While the naive implementation solves the problem in $O(n^2l)$, the program implements a radix search tree structure for binary keys based on [this paper](https://onlinelibrary-1wiley-1com-10000956e0040.eczyt.bg.pw.edu.pl/doi/full/10.1002/cpe.5027). A trie is initialized with all of the strings in $O(nl)$ time. Then, for each of the strings and each of the indicies i = $0, 1, ..., l - 1$ we can check if the trie contains a string that differs from the currently processed string on position i in $O(l)$. Thus, searching for each of the keys takes $O(nl^2)$ time. The algorithm is further accelerated by parallelizing the search for each bit position for a given string.

Program was implemented using C++/CUDA.