We enlist a two layer neural network which consists of one hidden layer and one output layer.
The following is almost the same as before in [[Bigram Model (Neural Networks)]] for initializing the dataset building lookup tables.

First, we initialize two lists `X` and `Y`.
```python
X = []
Y = []
```
*Note: As always X will be the feature and Y will be the label.*

In order to make the predictions more meaningful we bump up the prediction context window to `3` for this MLP i.e. 
`[e, m, m] -> a`
X will handle the `context` for the next prediction for `a` which is `e m m`.

Now we can hard-code this to be:
```python
context = [0 , 0 , 0]
```
*Note: We are going to pad the front of the context with '.'  that's why we initialize the context with zeros. As, stoi['.'] = 0.*

Since we want to change as little as possible if we ever want to scale it we opt to multiply the list by a `block_size`.

```python
block_size = 3 # Defines the size of context window.

context = [0] * block_size # Context window
```

Then we loop over the words to populate and build a dataset. Before we do that lets build the tiny helpers.
```python
# Helpers
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.item()}
```

```python
# Building The Dataset.
X = []
Y = []
block_size = 3

for w in words:
	context = [0] * block_size
	for ch in w + '.': 
		index = stoi[ch]
		X.append(context)
		Y.append(index)
		# Print Statement
		context = context[1:] + [index]
```

This loop builds the dataset. All we are doing in this loop is linking the context window to the corresponding label.

```python
print(''.join(itos[i] for i in context), '------>', itos[ix])
```
 
We can plug this above print statement in the loop to clearly see the building process.
