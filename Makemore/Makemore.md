	- Andrej Karpathy

```python
open('<filename>', '<mode>').read().splitlines()
```
Opens the file : `'names.txt'` with a mode: r

#### A quick overview of what we are going to do :
We are going to predict what letters/words follows the previous letter/word best.

So, for instance in "Emma" we get a frequency of 
'e followed by m' : 1
'm followed by m' : 1
'm followed by a' : 1

To add more context into this : we start the word with two unique tags
`<S> for start` `<E> for end`
This lets us take account more of the information from the word "Emma" like:
'Word starting with e' : 1
'Word ending with a' : 1

If we do this with even more words we can get a rough frequency calculation of how often does the corpus repeat some patterns of **certain letters being followed by certain letter/s** . With the unique values taken into account we can also count **how often certain letters are likely to start a word** as well as **how often certain letters are likely to end a word.**


The total number of frequencies can help us determine if a new pattern is likely to be the same for example if we construct a new word  starting with **E** (Say 'Emmy' this name has not yet been generated but for ease of understanding we will use it to be the result of this generation). 

#### A step by step approach (bigram)
- The word  starting from `E` has a possibility so it follows.
- `M` being the next letter after `E` also has  a higher frequency than other bi-grams and is possible. 
-  M being followed by M  again is also possible.
- M being followed by 'Y' hasn't been mentioned yet but they aren't impossible its just they don't occur frequently. 

But here you go a new name 'Emmy' has been generated using just a [[Bigram Model (Statistical Model)]].

Statistical models are more focused for interpretability, causal analysis, and structured data, which makes them less ideal for generative tasks like name creation. They do offer excellent transparency and work well with small datasets, they often fail to model the complex sequential patterns needed for fluent, realistic names. 

Deep learning approaches, on the other hand, are far more effective at automatically discovering intricate linguistic structures, resulting in significantly better generation quality. So, here we take a step towards [[Bigram Model (Neural Networks)]].

As we have small network and the data being fed to the [[Bigram Model (Neural Networks)]] is the same as the [[Bigram Model (Statistical Model)]] we come around the same metrics for the base variant of the neural net. that said the neural network model is still better as we can scale it to Tri-grams...N-grams. The statistical model follows a table/matrix populating approach this makes the modeling of greater combinations harder to train.

As an upgrade to the [[Bigram Model (Neural Networks)]] we implement a [[Multi-Layer Perceptron (Base)]] with  4 characters as inputs.










