A Bi-gram model is a language modeling technique that takes into account just its preceding letter/unique value to predict for it next letter.

For example, the tokens in a bigram model would look like this if we record for a word of "Hello".

We firstly add unique values to the start and the end in order to determine start and end of the word .
`Hello >> <S>Hello<E>`

Now, the bag of words will contains the tokens of this word.
```Bigram
<S> H : 1,
H  e : 1,
e  l : 1,
l  l : 1,
l  o : 1,
o <E>: 1,
```

The frequencies are counted across many names/words/sentences.

Since, there counts are going to happen across a lot of data, it provides us with some structured pattern of whats going on in this list of words,etc.

```python
# Making bigrams in a Pythonic way,
for w in words:
	for char1, char2 in zip(w,w[1:]):
		print(char1,char2)
```

In addition unique values are added in order to identify start and end:
```python
# Adding unique starts and ends to each word,
for w in words:
	chs = ['<S>'] + list(w) + ['<E>']
	for char1, char2 in zip(chs, chs[1:]):
		print(char1,char2)
```

Now, we are gonna learn the statistics/structure of the words and in Bigram Language Models counting the frequencies is one of the best method to model for structure.

# Using Dictionary
We will use a dictionary to maintain the counts of the bigram frequencies.
```python
# Initializing a Dictionary to keep track of bigram counts.
b = {}
for w in words:
   chs = ['<S>'] + list(w) + ['<E>']
   for char1,char2 in zip(chs, chs[1:]):
	   bigram = (char1, char2)
	   b[bigram] = b.get(bigram, 0) + 1	   
```
*A Dictionary is efficient as they don't register duplicates also has keys-values.

###### MINI EDA : 
```python
sorted(dict.items(), key = lambda kv: -kv[1])
```
This returns the most occurring bigram as the sorting key would be the value of index [1] of the returning dictionaries tuple. 

# Using Pytorch

It's a lot better if we use a 2D array in order to store the counts.
We will make use of pytorch tensors.

```python
import torch

a = torch.zeros((3,4))
a
#OUTPUS:
[[0,0,0,0],
 [0,0,0,0],
 [0,0,0,0]]
```

We can also change its datatype by passing a dtype= with the torch command.

Since, we are going to store every English alphabet and two special values.
The total size of the tensor shall be 26 + 2 = 28 i.e 28x28.

```python
# This will be out 2D Array of size 28x28 to store bigrams.
import torch
N = torch.zeros((28,28), dtype=torch.int32)
```

Then we shall populate it with the bigrams. But here comes a hurdle as we are working with the bigrams in a string format we aren't able to assign strings directly as tensor values so we somehow need to change them into integers. This is where a `lookup` table can help. 

*A Lookup table is like a key - value table where we enter a key and it outputs the value associated with the key.

```python
# We form a look-up table by enumerating charactes of the dataset.
chars = sorted(list(set(''.join(words))))
# this joins every words in the words dataset
# set() constructor just filter outs the duplicates 
# we actually need a list so we convert the structure into list
# then we sort everything out from a-z

stoi = {s:i for i,s in enumerate(chars)}
#enumerate function assigns key-type to each character a number in a `1 = char[0]` manner
# then 'string' : 'integer' lookup table is formed.

#NOTE: We need to manually add two special characters we made.

stoi['<S>'] = 26
stoi['<E>'] = 27
# This completes our look-up table
```


plt.imshow(N) is one way to display the 2D array but it looks kinda ugly so,

We come up with a reverse lookup table just to remap the integers into characters to display on the table.
```python
#Reverse Lookup Table
itos = {i:s for s,i in stoi.items()}
```

Plotting...
```python
#using matplotlib.pyplot as plt
plt.figure(figsize=16,16) # a canvas of 1600x1600
plt.imshow(N, cmap="Greys") #show a plot with colour map Grey
for i in range(28):
	for j in range(28):
		chstr = itos[j] + itos[i]
		plt.text(j,i,chstr,ha='center',va='top',color='teal')
		plt.text(j,i,N[i,j],ha='center',va='bottom',color='red')
plt.axis('off')
```

Now we see the 2D array more clearly. We can see that nothing can start with an ending tag `<E>` and we have whole row sitting at `zeros`.

In order to fix this we reduce two special values to one '.' value.
What this effectively does is it reduces the need of two `<S> a` and `<E> a` into a single element of `. a`. Which eliminates the column where the impossible words starting with an ending `<Ending> a`. This merges to the `<Starting> a` and as is the ending starters have a value of zero it doesn't affect the frequency count.
The loop now no longer checks for a value of (27,...)

### Update Code

```python
import torch
N = torch.zeros((27,27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
#Reducing Stoi
stoi = {s:i+1 for i,s enumerate(chars)}
stoi['.'] = 0

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1,ch2 in zip(chs,chs[1:]):
		bigram = (stoi[ch1],stoi[ch2])
		N[bigram] += 1

itos = {i:s for s,i in stoi.items()}

import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(N) # UGLY

plt.figure(figsize=16,16)
plt.imshow(N, cmap="Greys")
for i in range(27):
	for j in range(27):
		chstr = itos[j] + itos[i]
		plt.text(j,i,chstr,va="top",ha="center",color="teal")
		plt.text(j,i,N[i,j],va="bottom",ha="center",color="red")
plt.axis(off)
```

# Probability Distribution
`N[0] takes the first row of the 2D array, which convinently is our list of characters that start the word. ie <S> ..`

We take its probabilty distribution by converting it into float and by dividing it *Normalizing*  by its total sum. this way we get the probability of each `starting character`.

```python
p = N[0].float()
p = p / p.sum()
p
```

How to sample from these probability distribution ?
### We use `torch.multinomial` from `pytorch`

 `torch.multinomial(_input_, _num_samples_, _replacement=False_, _*_, _generator=None_, _out=None_) → LongTensor`
 
Returns a tensor where each row contains `num_samples` indices sampled from the multinomial (a stricter definition would be multivariate, refer to [`torch.distributions.multinomial.Multinomial`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.multinomial.Multinomial "torch.distributions.multinomial.Multinomial") for more details) probability distribution located in the corresponding row of tensor

Simple: I input various probabilities, It outputs integers according to the probability distribution.

```python
seed = torch.Generator().manual_seed(...)
index = torch.multinomial(p,num_samples=1,replacement=True,generator=seed)
itos[index.item()]
```

![[Pasted image 20260417031344.png]]

For efficiency purposes by not calculating the probability each time. We calculate a probability matrix. Also [[Broadcasting Semantics]] is heavily used,

```python
P = N.float() #Converting Data into float
P /= P.sum(dim=1, keepdim=True) # keeps data's dimension else squeezes it + Inplace operation
P
```
This generates 'P' a probability matrix that we can index into.

```python
for i in range(20):
	output = []
	index = 0 #starts with zero always
	while True:
		p = P[index]
		index =  torch.multinomial(p, num_samples=1, replacement=True, generator=g)
		output.append(itos[index])
		if index == 0:
			break
	print(''.join(output))
```

As we have a probability matrix as well, now we can map how much each bigram is likely to occur.
```python
for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1,ch2 in zip(chs,chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		prob = P[ix1,ix2]
		
		print(f"{ch1},{ch2} : {prob=:.4f}")
```




