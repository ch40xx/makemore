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
*Note: We are going to pad the front of the context with '.'  that's why we initialize the context with zeros. As, `stoi['.'] = 0`.*

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
itos = {i:s for s,i in stoi.items()}
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

```python
X = torch.tensor(X)
Y = torch.tensor(Y)
```

This loop builds the dataset. All we are doing in this loop is linking the context window to the corresponding label.

```python
print(''.join(itos[i] for i in context), '------>', itos[ix])
```
 We can plug this above print statement in the loop to clearly see the building process.

Following the footsteps mentioned in [[Benigo 2003]]. We are trying to make a MLP this requires us to define a **[[Embedding]]** vector to each of the input.

We initialize the embeddings as a shape `[27, 2]`. That means for each of our characters in our vocabulary we get 2 embedding. 
```python
vocab_size = len(itos)
n_embedding = 2

C = torch.randn((vocab_size,n_embedding)) 
```
*Note: `torch.randn` is used so the embeddings are randomized with a normal distribution.*

Here we embed the X's of our dataset / features using `C` the embeddings.

```python
x_emb = C[X] # x_emb.shape : [X.shape[0], X.shape[-1], 2]
```
X's value indexes into the C matrix to get the embeddings for that element in X.
*Thank you PyTorch !!!*

Let's take an example of `C[5]`.

```python
One-hot for index 5:              C (27 x 2):              Result:

[0, 0, 0, 0, 0, 1, 0, ..., 0]  @  [row 0: 0.2, -0.1]   = [row 5 of C]
  (1 x 27)                        [row 1: 0.8,  0.3]   = [-0.3, 0.9]
                                  [ ...            ]       (1 x 10)
                                  [row 5: -0.3, 0.9]
                                  [ ...            ]
                                  [row 9: 0.3, -0.5]
                                  [ ...            ]
```

The 1 at position 5 "selects" row 5 as everything else is zeroed out.
`C[5]` does the same thing instantly, without creating the one-hot vector.

	-Under The Hood.
```python
# E M M A
S = [[ 0,  0,  0],
	 [ 0,  0,  5],
	 [ 0,  5, 13],
	 [ 5, 13, 13],
	 [13, 13,  1]]
      
C = [[..1,..2],
	 [..3,..4],
	 [..5,..6],
	 [..7,..8],
	 [..9,..0],...]
	 
E = [
	 [[..1,..2],[..1,..2],[..1,..2]],
	 [[..1,..2],[..1,..2],[..9,..0]],
	]
```
*C is being indexed by S in the above mentioned way.*

Moving forward now we have in our hand x_embedded(`x_emb`). This is what we will work with as input for the rest of our network.

And to initialize a network we have to define a neuron.
A neuron is defined as: $$ Neuron : f\space (\space\sum_{i}\space w_ix_i + b \space)$$
*x : Inputs
w : Weights
b : Bias
f : Activation Function

Now we need to define **Weights** and **Bias**.

##### Weights
```python
W1 = torch.randn((block_size * n_embedding,n_neurons))
```

###### Why block_size * n_embedding is used as the neurons dimension?
This allows the model to capture the relationships between characters over the specified block size, enabling it to predict the next character effectively.[![16](https://external-content.duckduckgo.com/ip3/skeptric.com.ico)](https://skeptric.com/makemore-subreddits-part-4-backprop/makemore-subreddits-part-4-backprop.html)

##### Bias
```python
B1 = torch.randn((n_neurons))
```
*For each neuron we want a weight and a bias so this is initialized off the number of neurons we desire.*

### Hidden Layer Forward Pass

The first step in this layer would be to find the product of the `x_emb` and `W1`.


But we need to flatten the embeddings before the linear layer.
### Why?

Our context window is 3 characters: `e m m`. Each character gets looked up in the embedding table `C` and comes out as a small vector, say of size 2:

```python
e  →  [0.2,  0.5]
m  →  [-0.3, 0.9]
m  →  [-0.3, 0.9]
```

So your `x_emb` for this one example has shape `[3, 2]` — three characters, each with 2 numbers.

Now here's the key question: **what should the hidden layer receive?**

If we feed each embedding _separately_, the neuron looking at `e`'s embedding would have no idea that `m` and `m` came after it. It would be blind to the context. It would only ever see one character at a time, which makes it no better than the bigram model.

The whole point of the MLP upgrade was to give the model _more context_. So to hand the network all three embeddings _at once_, as a single input vector:

```python
# Flatten It
[0.2, 0.5, -0.3, 0.9, -0.3, 0.9]
       ↑ e       ↑ m        ↑ m
```

Now a single neuron in `W1` can have a weight for _each position_, meaning it can learn things like: "when position 1 looks like `e` AND position 2 looks like `m`, fire strongly." That's what learning character interactions means. The flattening is what _enables_ the hidden layer to reason about combinations, not just individual characters.

The `view(-1, block_size * n_embedding)` is just the mechanical step that performs this flattening. The _why_ is that you're giving the network the full picture at once.

Also, recalling what we previously had,
The shape of  `x_emb` is `[ batch_size, context_size, embedding_size]` i.e. `[5, 3,2]`.

According to matrix multiplication, we cannot multiply `[5, 3, 2]` @ `[6, 100]`. also, the `6` was inferred by multiplying the `block_size` with `n_embeddings` which happens to be the dimensions of our `x_emb` matrix. [Why ?](#Why-block_size-*-n_embedding-is-used-as-the-neurons-dimension?)

Now there are several ways to reshape this matrix to fit the `W1` matrix.
Let's try this manual method for the tensor below, for example:

```python
tensor([[[-1.1, -1.8],[-1.1, -1.8],[-1.1, -1.8]],
        [[-1.1, -1.8],[-1.1, -1.8],[ 0.5, -0.4]],
        [[-1.1, -1.8],[ 0.5, -0.4],[ 0.1,  0.8]],
        [[ 0.5, -0.4],[ 0.1,  0.8],[ 0.1,  0.8]],
        [[ 0.1,  0.8],[ 0.1,  0.8],[ 1.0,  2.1]]])
```

Here If we remove `#` dimension from the `[:, #, :]` we see that it causes the matrices to reorder such that the `3` splits are merged into the single dimension.  So, the each dual grouping of elements by the 2nd dimension are clumped together.

```python
tensor([[-1.1, -1.8, -1.1, -1.8, -1.1, -1.8],
        [-1.1, -1.8, -1.1, -1.8,  0.5, -0.4],
        [-1.1, -1.8,  0.5, -0.4,  0.1,  0.8],
        [ 0.5, -0.4,  0.1,  0.8,  0.1,  0.8],
        [ 0.1,  0.8,  0.1,  0.8,  1.0,  2.1]])
```

There are several methods to achieve this:
1. Concatenation Method

- Manual Method
```python
torch.cat([x_emb[:,0,:],x_emb[:,1,:],x_emb[:,2,:]],dim=1)
```

```python
print(x_emb[:,0,:],x_emb[:,1,:],x_emb[:,2,:])
#Elements of [3,0]         
(tensor([[-1.1, -1.8],[-1.1, -1.8],[-1.1, -1.8],[ 0.5, -0.4],[ 0.1,  0.8]]),
#Elements of [3,1]         
 tensor([[-1.1, -1.8],[-1.1, -1.8],[ 0.5, -0.4],[ 0.1,  0.8],[ 0.1,  0.8]]),
#Elements of [3,2]         
 tensor([[-1.1, -1.8],[ 0.5, -0.4],[ 0.1,  0.8],[ 0.1,  0.8],[ 1.0,  2.1]]))
```

- Unbind Method
```python
torch.cat(torch.unbind(x_emb,1),1)
```

This method comes with it cons as it generates and stores are new tensor in the memory.

2. View Method

PyTorch store every tensor in a one dimension row vector.
```python
x_emb.storage()
```

```python
preact = x_emb.view(-1, block_size * n_embedding) @ W1 + B1
```
Learn more about [torch.view()](https://www.geeksforgeeks.org/python/how-does-the-view-method-work-in-python-pytorch/)

##### Activation Function
```python
f = torch.tanh(preact)
```
 To add some non-linearity to the model we need an activation function. For the hidden layer activation function we have chosen `torch.tanh()`.

### Output Layer Forward Pass

*f*  is now transferred over to the output layer, in order to count for the log-counts.

Lets define the output layer's:

```python
W2 = torch.randn((n_neurons,vocab_size))
B2 = torch.randn((vocab_size)) # Note: Bias should match output dimension
```

Then we calculate the log-counts:
```python
logits = f @ W2 + B2
```

For the activation function, we are going to use softmax.

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True) # Note: sum over dimension 1
```

At last, for the loss function we are going to use, **negative log likelihood**.

```python
loss = -(probs[torch.arange(X.shape[0]), Y]).log().mean()
```

We first get the probability for each of the labels of Y and using a negative log likelihood we calculate the mean of the likelihood then we try to get it closest to 0 so the probability goes up to 1 or close enough.

Here comes the great **`back propagation`.**
```python
loss.backward()
```

*Note: We have to initialize the gradients to zero every time we update the weights. so that the gradients don't get accumulated.*


##### Updating the Parameters.

We have several parameters till this point:

```python
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2),generator=g)
W1 = torch.randn((6,100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100,27), generator=g)
b2 = torch.randn(27,generator=g)
parameters = [C, W1, b1, W2, b2]
```

Before the executing the back propagation.
We have to turn the `requires_grad=True`.

Then the gradients must be turned to zero.
```python
for p in parameters:
	p.grad = None
# Then we do,
loss.backward()
```

For updating the parameters:
```python
for p in parameters:
	p.data += -0.01 * p.grad
```
 
 Here, the `0.01` is the [[Learning Rate]] value. we use the (-) sign to always reduce the loss. Learning Rate is extremely important as it quite literally determines at what rate the model learns the features. 
 
 *Note: Too much learning rate can introduced noise in the model, while too little can be quite slow for the model to learn the features.*

# Mini-Batch Construction

We are using the whole dataset for the forward pass. This can result in a heavy load to the computer when training in large datasets. In order to make it effective yet fast we train the data in batches, so the training time reduces for each run.
```python
torch.randint(low=0,high=X.shape[0],size=(batch_size,))
```

We then get a `1D` tensor of `batch_size = 43` randomly chosen integers chosen integers to be the inputs for the embedding matrix.

# TRAIN, VAL/DEV, TEST SPLIT

It is also not the best idea to let the model learn from every single item from the dataset. We need to split it to test its real metrics.

So, we split this into 3 portions:
**Train :** *80%*
**Val :** *10%*
**Test :** *10%*

We use the **80% Training Split** of the data to train the model
We use the **10% Validation Split** to check the accuracy and other metrics of the model this is still largely unseen by the model but can be used to infer proper metrics.
We use the **10% Test Split** to check the final metrics of the model. Usually this is ran only once to keep the model from learning from unseen data.

```python
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev= build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```

With this we can now use the training set along with the mini batches to train the model.
```python
ix = torch.randint(0, Xtr.shape[0], (32,))
    
# Forward Pass
x_emb = C[Xtr[ix]]
h = torch.tanh(x_emb.view(-1,6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ytr[ix])
print(loss.item())

# Back Propagation
for p in parameters:
	p.grad = None
loss.backward()

# Update
learning_rate = 0.001 if i < 50000 else 0.0001
for p in parameters:
	p.data += -learning_rate * p.grad
```
*Note: Here we train the model while keeping track of the each mini-batch loss.*

```python
# Loss Calculation For Whole Training SET
x_emb = C[Xtr]
h = torch.tanh(x_emb.view(-1,6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Ytr)
print(loss.item())

# Loss Calculation For Unseen Validation SET
x_emb = C[Xdev]
h = torch.tanh(x_emb.view(-1,6) @ W1 + b1)
logits = h @ W2 + gb2
loss = F.cross_entropy(logits,Ydev)
print(loss.item())
```

