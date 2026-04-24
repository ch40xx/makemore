We first start with a 
a feature list : `xs`
And 
a label list : `ys`

```python
xs = []
ys = []
```

then we iterate over each of the label and assign the preceding character as the feature then the character to be predicted as a label.
This way we have a dataset of what needs to follow what in such a sense that the model can learn something using this dataset of patterns.

```python
for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1,ch2 in zip(chs,chs[1:]):
		# since we cant store strings directly we pass it through a lookup table to get the corresponding integer
		ix1 = stoi[ch1]
		xs.append(ix1)
		ix2 = stoi[ch2]
		ys.append(ix2)
```

For example,
in `.emma.`
`xs` will contain `[., e, m, m, a]`
`ys` will contain `[e, m, m, a, .]`

The model shall learn if it encounters `xs[0] i.e. '.'` there shall be a very high probability of `ys[0] i.e 'e'` being the next character.

after this to makes things easier for us and to take advantage of PyTorch library we convert these lists into tensors.
```python
xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

*Note: tensor vs Tensor : in torch.tensor you can change the dtypes later while in torch.Tensor you're kinda stuck with float.*

This is almost ready to be fitted into a model for predicting next characters. There we usually convert this into vector features such as one_hot vector in this case. The model doesn't learn with 0,5,13,13,1,etc plotted. *Basic Machine Learning*

To, one_hot encode this feature we use *torch.nn.functional.one_hot()* :

```python
import torch.nn.functional as F
xenc = F.one_hot(xs,num_classes=27).float()
```
We can also define how many classes we need in our one hot encoder. Since we prefer float based operations over integers we convert this into float to pick up the precision nuisances of the trailing decimal numbers.

	- RECAP
	1. We know what a neurons structure looks like i.e input(x), weight(w) and bias(b) passing through an activation function.
	2. We have just defined the input(x) as xenc-x encoded in one_hot.
	3. We need to find a weight to multiply with the input.
	4. We also need to define an activation function to pass the products through.

$$ Neuron = x*w + b  | softmax $$

For the weights, we take random normal distribution of numbers using PyTorch `torch.randn`.

```python
W = torch.randn((27,27), generator=g) 
```

*Generator=g just to make it deterministic.*

We are going to matrix multiply these two resulting in logits: log counts.
While matrix multiplication happens we can visualize it as the element that is turned on during one hot encoding picks up the Weight from the `W` matrix and rest of the other is squashed as they are just ones and it sums up anyway. This closely resembles [[Bigram Model (Statistical Model)]] method of count of the bigrams, while this is much smaller than the count in this method the weights do the work of being randomly assigned but in the later phase as we deploy back propagation these weight come to be aligned near the "counts of bigrams" as we approach minimum loss.

# Forward-Pass

When the `W` reaches a close enough point the the same one hot encoded row picks the weight from W and squashes the rest while summing.

```python
logits = xenc @ W
```

Here the logits are very small numbers sometimes negative as well. In order to eliminate this discrepancy and also as a crucial method of the softmax function. We will exponentiate the logits.

$$ Softmax \space | \space(\sigma(\vec{z})_i) = \frac {e^{z_i} } {\sum_{i=1}^k e^{z_i}} $$

$$
z_i : logits

$$


```python
counts = logits.exp() # Equivalent to N of the statistical model
probs = counts / counts.sum(1,keepdims=True)
```

The softmaxxed value of our product of input and the weight will serve as the probability matrix. This is also called the **forward pass**, for the neural network  version of the bigram character level language model.

We can check how the probability is being assigned for each of the desired label by manually indexing into the position.

`probs[0,5]` the probability of the first letter being 'e', etc.

We can see that is not doing its best as the loss must be very high, after all its just  being modeled after the random weights that were generated and like every neural network, we always adjust the weights inside the neuron. Inputs cant be changing, the only two variables are weights and biases. since this model doesn't have a bias fitted into it we can only adjust the weights to reduce the loss.

We need a way to calculate the loss then back propagate from the loss ultimately finding its way back into weights which helps us see how much each element of the weight is having what kind of influence on the final loss.  This lets us determine how the weights shall be adjusted to best fit the loss reduction.

Log likelihood once again comes in handy to check for the loss in our model.
Now we need to check over one example to fix the weights of the whole model.

```python
loss = probs[torch.arange(5),ys]
```

loss shall be the probability of the input and the desired labels. from out log likelihood breakdown in previous statistical model. we need to `torch.log()` in order to limit the output range. 
```python
loss = probs[torch.arange(5),ys].log()
```

many probabilities would be negative so we make it positive by multiply it with a negative sign. this is called the negative log likelihood as previously mentioned.

```python
loss = -probs[torch.arange(5),ys].log()
```

We tend to take the normalized from of the negative log likelihoood.
So,

```python
loss = -probs[torch.arange(5),ys].log().mean()
```

# Backward Pass

Before we commit to the backward pass we need to fix few things such as

```python 
W = torch.randn((27,27),generator=g,requires_grad=True)
```

After this W can store up gradients.

```python
W.grad = None # To clear out gradients and eliminate accumulation
loss.backward()
```

# Update Weights

```python
W.data += -0.01 (learning rate) * W.grad
```

W.data hold the weight's data but the learning rate producing with the gradient and adding these to the data we reduce the loss.

The #forward-pass, #backward-pass and the #update-weights working together helps to reduce the loss and get the best `W i.e weights matrix` for inference.

# Inference / Sampling

```python
for i in range(5):
	out = []
	ix = 0
	while True:
		xenc = F.one_hot(torch.tensor(ix),num_classes=27).float()
		# We have a good W matrix as a result of training
		logits = xenc @ W # log counts
		counts = logits.exp() # Softmax I - N equivalent
		probs = counts / counts.sum(1,keepdims=True)
		ix = torch.multinomial(probs,num_samples=1,replacement=True,generator=g)
		out.append(itos[ix])
		if ix == 0:
			break
	print(''.join(out))
```

Everything is quite the same except:

1. `xenc = F.one_hot(torch.tensor(ix),num_classes=27).float()`
since we are always going to use the index as `0` to start sampling from as it contains the starting `. +  ...`.

2. `torch.multinomial(probs,num_samples=1,replacement=True,generator=g)`
   It plucks out an integer based on the probabilities we give to it.

This concludes our neural network model for bigram language model.
