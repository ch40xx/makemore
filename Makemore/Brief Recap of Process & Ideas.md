In [[Multi-Layer Perceptron (Base)]], we initialized two resulting tensors, that are populated by the words with a  context window of `three` characters. Then, we split it into three parts of : `Train`, `Val`& `Test` . This lets us train the model with larger portion of data while we test on unseen validation data and at last confirm the metrics if they are suitable using test data.

We implement a network with one Hidden layer and one Output layer. The blocks of content i.e features`( e m m )`are first converted into integers as we cannot model it directly from string values.

	Previously, we had used `one-hot encoding` in [[Bigram Model (Neural Networks)]] to index into weights which was equivalent of indexing the counts of the `Count Matrix`. The only difference in the neural network version of the Bigram Model is the Weights are initialized at  random and they serve as a `Random Count Matrix` during the first of the training phase.

Coming back to the point we took things different with the [[Multi-Layer Perceptron (Base)]] approach by using [[Embedding]]'s Matrix. In our use case this lets us further provide the context about the characters to the model without using a `Randomly Initalized Weights` or a dimensionally cursed `Count Matrix`.
*Basically, we are using the embedding matrix to turn the feature into feature vectors.* For the example of [[Multi-Layer Perceptron (Base)]], this has a embedding value of `2`  for each character so the shape is `[27,2]` which lets each of the character obtain 2 of the embedding when indexed. 

	Now, we had been indexing into the Embedding Matrix, say `C` right  away then implementing the forward pass. We slightly delay it here as we add minibatches to the training as well as remember we previously had split the dataset into three groups of Train,Val & Test	{Xtr, Ytr : Feature-Train, Label-Train}, {Xdev,Ydev : Feature-Val, Label-Val}, {Xts,Yts : Feature-Test, Label-Test} respectively. So we need keep those in mind.

During the previous [[Bigram Model (Statistical Model)]], [[Bigram Model (Neural Networks)]] and also the first phase of [[Multi-Layer Perceptron (Base)]]. We had been just initializing the Layers of the network when needed then carrying out the forward pass and backward pass. This time we create the required parameters separately in a respected manner which also lets us make changes to the parameter more easily as we  need  to sort out some **initialization inefficiencies** we will talk about in a bit. So now we initialize the parameters.

```python
g = torch.Generator().manual_seed(123456) #Making it determnistic.

#---Hidden Layer Parameter---
W1 = torch.randn((block_size * emb_count, n_neurons),generator=g)
B1 = torch.randn((n_neurons), seed=g)\

#---Output Layer Parameter---
W2 = torch.randn((n_neurons,vocab_size), generator=g)
B1 = torch.randn((vocab_size), generator=g)

parameters = [g, W1, B1, W2, B2]
```

Lets talk about the parameters here and some whys in brief.

---
**`W1`** : It is the **weights for the hidden layer**, its column shape is `(block_size * emb_count)` as we are going to flatten the `Indexed embeddings` and its gonna result in a `row vector` containing `6` elements.  

**How ?** : Because as we are taking a `context window` of  `3` characters and each character will be getting 2 `embedding` for each character. 
```python
[('e' -> [ emb1, emb2]),
 ('m' -> [ emb1, emb2]),
 ('m' -> [ emb1, emb2])]
```
We will unbind it from the shape of `[1,3,2]` into the shape of `[1,6]` and we can do it in few ways using PyTorch.  See [[Multi-Layer Perceptron (Base)]].

**Why ?**: We need to flatten it as we need to serve the context of the whole 3 `block_size` context window to the neuron instead of just one character. 
	*Visualized :*
	'e' -> {Neuron} -> No previous context
	'e' 'm' -> {Neuron} -> Bigram Model just complexified, no quality gain
	'e' 'm' 'm' -> {Neuron} -> Three embedding  sent together, holds context and lets us generate better results.

And,  following the PyTorch broadcasting the shape shall match in order to perform operations on the two matrices.

---

**`B1`** : It is the **biases for the hidden layer**. It gonna be added to each of neuron so its shape is `[1,n_neurons]`.

---

**`W2`** : It is the **weights for the output layer**, its column size is `[n_neurons]` and the row size is `[vocab_size]` its because we need to converge, we are  getting a `100 element` outputs and it shall converge to `27 elements`.

---

**`B2`** : The `biases for the output layer`. Each gets added to the output from the dot product multiplication of output layer weights (`W1`) and hidden layer output   (`h`) .

---

The parameters are set inside a list so we can access these in bulk, which will be handy later.

Following the tutorial we initialize the hidden layer first. *We only define two layers for this network.* 

As we know the forward pass till now consists of :
1) Embedding the inputs
2)  then, `inputs @ W1 + b1` - this gives us the linear activation
3) to introduce non-linearity to the model we use an activation function,
	-  `sigmoid()`
	- `tanh()`
	- `relu()`
	- They all squash the larger number-line we get into a constrained environment.
	the activated hidden layer output is now multiplied  and added `[standard matrix multiplication]` i.e `h @ W2 + b2`.
-  Now we put it thru the output layer activation and we use softmax.
-  For the loss function, we use negative log likelihood as always.


#### Forward Pass

While building [[Multi-Layer Perceptron (Base)]], we introduce mini-batches to make the computation easier. Rather than using the whole dataset to train at once, we take random number of examples in batches then train on it.
To add on top of this we use the `Training Data Split.`

```python
indexes = torch.randint(low=0, high=Xtr.shape[0], (batch_size=32,), generator=g)
```
We take 32 elements randomly from the lowest 0 index to highest which is the column shape of the training data.

```python
Xb, Yb = Xtr[ix], Ytr[ix]
```
We define two working Xb and Yb.

```python
xemb = C[Xb]

# Hidden Layer Pass
hpreact = xemb.view(-1,(block_size * emb_count)) @ W1 + B1
h = torch.tanh(hpreact)

# Output Layer Pass
logits = h @ W2 + B2 # ~ z
"""
Yeah we use softmax followed by nll to calculate the loss.
---Softmax---
counts = logits.exp() ~ z.exp() 
probs = counts / counts.sum(1, keepdim=True) ~ z.exp() / sum of z.exp()
loss = -probs[torch.arange(Xb.shape[0]),Yb].log().mean()
--------------
We can use this or use something much easier, Thanks PyTorch Again.
"""
import torch.nn.Functional as F
loss = F.cross_entropy(logits,Yb)
print(loss.item())
```

`Cross Entropy Version` is a much better method of calculating the loss as the traditional way can introduce some bugs in a few ways that may go unnoticed.

For base MLP, Final Test Loss:
![[Pasted image 20260509114353.png]]
#### Backward Pass

For Inference,
We are still sampling using `softmax` for setting the probabilities and `torch.multinomial` is used to take items form the probabilities. then convert them into letters using the int to strings lookup table.

We get some decent samples of name.

![[Pasted image 20260509114156.png]]

Pretty decent names, sorta-name like names compared to early models.

![[Pasted image 20260510171352.png]]

But here is the catch we can make it even better. See [[Multi-Layer Perceptron (Optimized)]].















