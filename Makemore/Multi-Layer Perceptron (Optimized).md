Making the neural network better. Lets look at the problems.
### 1. One of the key pain points is the hockey stick graph.

Lets look at the initial loss our model has computed.
![[Pasted image 20260510172919.png]]
We get a initial loss of **(51.8)**

As you can see we are wasting some of our precious time and resources trying to get to a suitable loss to start training with. 

![[Pasted image 20260510172538.png]]
Whats happening is we are starting with a very high loss at start then we drop down quick.

Over `1000 Iterations` of training with a learning rate of `0.1` 
We see a loss of `8.9` (for that mini batch). But even in the graph we see a high loss at the start.

Lets calculate an ideal loss/ suitable loss we would like the model to start with.
If say we give each letter  an equal probability. then its gonna be `1/27 ~ 0.037`.
Since we know the probability, we can get its loss by using negative log likelihoog and its around `3.29`.
![[Pasted image 20260510215318.png]]

#### Why does this happen ?
As we are initializing the parameters randomly through `torch.randn()` we get randomly assigned values that populate the tensor. These values can sometime vary greatly, we need something closer to zero for initialization, something closer for entropy or something zero.

Lets take an example of how the random numbers affect our losses.
```python
logits_ex = torch.tensor([0.0, 0.0, 0.0, 0.0])
probs_ex = torch.softmax(logits_ex, dim=0)
loss = -(probs_ex).log().mean()
print(logits_ex, probs_ex,loss)
```
*Note: Softmax is expotentiating the logits then adding them to divide the exponentiated logits.*

Remember we want the initialization to be close to zero and Gaussian. so if the initializations record high amount of numbers randomly the losses can start with high numbers too.

To minimize this slight error in our math what shall we do ? Lets see we are currently getting the logits by doing `h @ W2 + B2`. We need the weights's initialization to be near zero so that the h takes on values closer to 0. and we can effectively initialization the `B2` to be zero for the mean time as its just adding up.

This lets us record something close to what we regard as a good loss instead of the previous **`51.8`**. Now, we get **`3.38`** just by
```python
W2 = torch.randn((579,27), generator=g).to(device) * 0.01
B2 = torch.randn((27), generator=g).to(device) * 0
```
setting these to near `0`.

Even here the Training and Validation Dataset show minimal loss than before,
![[Pasted image 20260511145302.png]]
As we are spending more time actually optimizing the network instead of squashing the randomly initialized weights to be closer to zero.

Here, the ideal we want is **`3.29`** so lets make some changes to achieve those.
### 2. Another is the initialization randomness.

We have to take a look at the activation values (hidden states).
![[Pasted image 20260511150025.png]]

See that the values are mostly `1` and `-1`, This is because we are using `tanh()` as the activation function and any values that are very high gets squashed to `1` and any values that are very low get squashed to `-1`. 

This hints us that the values we are getting after the neuron calculates `x @ W1 + B1` also has flaws within itself a.k.a they are being initialized at extreme values rather than close to `0`.

Lets see if that's true;
- Checking the `tanh()` activation:
	![[Pasted image 20260511151849.png]]
	We see that the if its true then it turns white so most of the values are triggering the `tanh` activation due to them being very high values.
- Checking the activation values:
	![[Pasted image 20260511152725.png]]
	Most of the activation values result in -1 or 1.
	
- Checking the pre-activation value Distribution:
	![[Pasted image 20260511152400.png]]
	We  can see that the values are way off from the center mostly taking values above 0 in both sides.

These distributions are wrong and inefficient.

So, what do we do ? Lets try making the Weights and Biases for the hidden layer closer to 0 like we previously did with the output layer parameters.

```python
W1 = torch.randn((51,579), generator=g).to(device) * 0.01
b1 = torch.randn((579), generator=g).to(device) * 0
W2 = torch.randn((579,27), generator=g).to(device) * 0.01
b2 = torch.randn((27), generator=g).to(device) * 0
```

This works as the initial values are close to zero and they are already in a Gaussian thanks to `torch.randn()`. 

Now lets check the distributions.
- Checking the `tanh()` activation:
	![[Pasted image 20260511152958.png]]
	We see that the if its true then it turns white so most of the values are not triggering the `tanh` activation due to them being multiplied by our manual zeros.
	
- Checking the activation values:
	![[Pasted image 20260511153106.png]]
	Now, none of the activation values result in -1 or 1. The values are capped around `-0.4` and `0.4`.
	
- Checking the pre-activation value Distribution:
	![[Pasted image 20260511153226.png]]
	
	We  can see that the values are closer to the center and mostly taking values around 0 in both sides. In fact the `tanh()` activation isn't activating anything we can check the graph from above.

Okay, that worked. Now lets check the losses.
The first ever loss we record is `3.29`. and the training and validation loss are `3.26` and `3.26` respectively. This is a good place to start.

We are no longer stuck with that hockey stick graph. And most of our initialization values are close to zero and our precious training time is used to optimize the neural network instead of reworking the values.

That might have worked, i mean the principle is same that it needs to be `0.0` but there are  certain ways to do it. ALSO, lets look at what happens to the Gaussian distribution when we perform operations.

![[Pasted image 20260511210038.png]]
We can see that the standard deviation changes when the two multiply together, this means that the Gaussian values wants to spread. and the deviation gets bigger.

If we multiply the weights with larger numbers the std gets even bigger.
![[Pasted image 20260511210258.png]]

There's a way to preserve the Gaussian difference. and it is to divide the multiplying factor by the square root of `fan_n` ~ `the number of inputs`.
![[Pasted image 20260511210546.png]]
With these value we preserve the Gaussian spread.

This is talked about in the paper `Delving Deep into Rectifiers` by `Kaiming He et al.` They found out that the when they are squashing the functions they need to add certain gain according to the activation function they are using.
So this became the standard to multiply the weights with the standard deviation of $$std ={\frac{gain}{\sqrt{fan_n}}}$$
in order to keep them in unit deviation.

Here's a PyTorch table to see the different gains for different activation used.
![[Pasted image 20260511235746.png]]

So our initialization would be like this,
```python

W1 = torch.randn((51,579), generator=g).to(device) * (5/3) / (51**0.5)
b1 = torch.randn((579), generator=g).to(device) * 0
W2 = torch.randn((579,27), generator=g).to(device) * 0.01
b2 = torch.randn((27), generator=g).to(device) * 0
```

Since we have solved the problems lets try to make them easier to handle with bigger and deeper networks. We solved our problems by `multipliying the values with 0...ish numbers` or some number which came with a principled approach of the research paper[Kaiming He et al.] which is not incorrect but very painstakingly hard when paired with a deeper network. This can introduce bugs and also is time consuming.

Andrej mentions that it was very important to initialize the network very properly in the early days of making neural networks, but now there are some modern innovations that lets us be somewhat carefree with the initializations.

One being **Batch Normalization** discussed in the paper of **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** by `Sergey Ioffe` and `Christian Szegedy`. What they came to find is if we need the weights to be in unit deviation and roughly Gaussian why don't we take the hidden values and make them Gaussian with unit deviation instead.
![[Pasted image 20260512001051.png]]
We do this by:
- Getting the mini batch's mean.
	I get confused as Pytorch transposes the text book visualization of the layers but its like this we take the mean of all the values in the neuron.
	Using, `minibatch.mean(dim=0, keepdim=True)`
	What this does is we take every example that fires that neuron and get a mean from it.
- Then, We can calculate the standard deviation to normalize it.
	Here we don't need to calculate the variance as PyTorch lets us get the `std`. But the in variance we are just calculating the each inputs difference with the mean of the batch then dividing it by the number of inputs. and we can square root that to get the standard deviation. so it seems like we skipped to normalizing step.
- Finally, we subtract the values with the mean and divide it with the standard deviation. $$ \hat{x}= \frac{x_i - \micro_\beta}{\sqrt{\sigma^2_\beta + \epsilon}}$$ We can perfectly differentiate this,which means we can train this. 
 ```python
 # So no we change the Optimization Steps Slightly to Introduce 
 # Batch Normalizaiton.
 max_steps = 200000
 batch_size = 51
 lossi = []
 
 for i in range(max_steps):
	 # MiniBatch Construct
	 ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
	 Xb, Yb = Xtr[ix],Ytr[ix]
	 
	 # Forward Pass
	 emb = C[Xb]
	 preacth = emb.view(-1,1) @ W1 + B1
	 hmean = preacth.mean(0, keepdim=True) #batch mean
	 hstd = preacth.std(0, keepdim=True) #batch std
	 preacth = (preacth - hmean) / hstd
	 h = torch.tanh(preacth)
	 ...
 ```
But this wont do very well as we need this to be Gaussian at initialization only not every time the loop runs so we will need to introduce something that will be training and the model figures out the rest after initialization.

- That is, **Scale and Shift**.$$y_i = \gamma\hat{x}_i + \beta$$
	The final output will be we multiply some batch-gain and add some batch-bias to the normalized inputs/preacth. These will be trainable.

```python
# Parameters
W1 = torch.randn((51,579), generator=g).to(device) * (5/3) / (51**0.5)
b1 = torch.randn((579), generator=g).to(device) * 0
W2 = torch.randn((579,27), generator=g).to(device) * 0.01
b2 = torch.randn((27), generator=g).to(device) * 0
bngain = torch.ones((1,579),generator=g).to(device)
bnbias = torch.zeros((1,579), generator=g).to(device)
```

 ```python
 # So no we change the Optimization Steps Slightly to Introduce 
 # Batch Normalizaiton.
 max_steps = 200000
 batch_size = 51
 lossi = []
 
 for i in range(max_steps):
	 # MiniBatch Construct
	 ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
	 Xb, Yb = Xtr[ix],Ytr[ix]
	 
	 # Forward Pass
	 emb = C[Xb]
	 preacth = emb.view(-1,1) @ W1 + B1
	 hmean = preacth.mean(0, keepdim=True) #batch mean
	 hstd = preacth.std(0, keepdim=True) #batch std
	 preacth = (bngain * (preacth - hmean) / hstd) + bnbias
	 h = torch.tanh(preacth)
	 ...
 ```

This kinda doesn't change it right now but its gonna make it easier for us to train deeper networks as we don't have to manually go in and normalize the weights and biases for each layer. instead we introduce a batch normalization for each batch that lets us normalize the layer automatically.

From this training, when in inference this expects us to input batches for inference as the statistics are being calculated as batches instead of the whole dataset. So no in order for us to infer from single inputs we need to do something with the statistics at the time of inference.

```python
# after the training we will calculate the mean and the standard deviation of the whole dataset
with torch.no_grad:
	emb = C[Xtr]
	embcat = emb.view(emb.shape[0],-1) 
	preacth = embcat @ W1 + b1
	bnmean = preacth.mean(0, keepdim=True)
	bnstd = preacth.std(0, keepdim=True)
	# preacth = bngain * (preacth - bnmean) / bnstd + bnbias
```

Now we've calculated it we can now see the losses as follows.
```python
@torch.no_grad()
def split_loss(split):
	x,y = {
		'train': {Xtr,Ytr},
		'dev': {Xdev,Ydev},
		'test': {Xte,Yte}
	}[split]
	emb = C[x]
	embcat = emb.view(emb.shape[0],-1
	hpreact = embcat @ W1 + B1
	hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
	h = torch.tanh(hpreact)	
	logits = h @ W2 + B2
	counts = F.cross_entropy(logits,y)
	print(split, loss.item())
	
split_loss('train')
split_loss('val')
```

A better way to calculate the mean and std of the full dataset is by keeping track of the whole mean and std. we can do this by introducing two buffers in the parameters.

```python
# Parameters
C = torch.randn((27,3))
W1 = torch.randn((51,579), generator=g).to(device) * (5/3) / (51**0.5)
#b1 = torch.randn((579), generator=g).to(device) * 0
W2 = torch.randn((579,27), generator=g).to(device) * 0.01
b2 = torch.randn((27), generator=g).to(device) * 0

bngain = torch.ones((1,579),generator=g).to(device)
bnbias = torch.zeros((1,579), generator=g).to(device)
bnmean_running = torch.zeros((1,579), generator=g).to(device)
bnstd_running = torch.ones((1,579)), generator=g).to(device)

parameters = [C, W1, W2, b2, bngain, bnbias]
```

We are going to calculate this on the side while training the network.

```python
 # Batch Normalizaiton.
 max_steps = 200000
 batch_size = 51
 lossi = []
 
 for i in range(max_steps):
	 # MiniBatch Construct
	 ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
	 Xb, Yb = Xtr[ix],Ytr[ix]
	 
	 # Forward Pass
	 emb = C[Xb]
	 preacth = emb.view(-1,1) @ W1 #+ B1
	 
	 bnmeani = preacth.mean(0, keepdim=True) #batch mean
	 bnstdi = preacth.std(0, keepdim=True) #batch std
	 preacth = (bngain * (preacth - bnmeani) / bnstdi) + bnbias
	 with torch.no_grad:
		bnmean_running = 0.999 * bnmean_runnning + 0.001 * bnmeani
		bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
		
	 h = torch.tanh(preacth)
	 ...
```

This lets us use the bnmean_running and bnstd_running directly in the inference.
```python
@torch.no_grad()
def split_loss(split):
	x,y = {
		'train': {Xtr,Ytr},
		'dev': {Xdev,Ydev},
		'test': {Xte,Yte}
	}[split]
	emb = C[x]
	embcat = emb.view(emb.shape[0],-1
	hpreact = embcat @ W1 #+ B1
	hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
	h = torch.tanh(hpreact)	
	logits = h @ W2 + B2
	counts = F.cross_entropy(logits,y)
	print(split, loss.item())
	
split_loss('train')
split_loss('val')
```
*You can not use the` b1 `as its basically useless at this point as the bias gets subtracted and the batch norm bias will take over as the new bias.*
