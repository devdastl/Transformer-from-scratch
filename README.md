# Repository to learn Transformer architecture with Andrej
This is a repository with includes hands-on implementation of transformer architecture.

## Dataset used.
This repository uses tiny shakespeare dataset [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## Model architecture
We created multiple model architectures to reach transfomer:

#### Simple Bigram model-V1:
It uses just one embedding matrix of size (vocab_size, vocab_size) and then outputs logits.

#### Simple Bigram model-v2:
It uses embedding matrix, positional embedding and linear layer to output final logits.

#### Simple Bigram model-v3:
With addition to all the layers from model-V2, this one has single attention head implemented.

#### Transformer:
This is the final version with following additional features compared to model-V3:
1. Implementaion of MultiHead attention.
2. Implementation of feed-forward layer.
3. Implementation of Transformer block chaining multiheaded attention and feed-forward layer.
4. Added skip connection before multiheaded attention and before feed-forward layer. For better optimization.
5. Added layer normalization and dropout.

## Notebook experiments:
Here is the link of the notebook where experiements are done [colab notebook](https://colab.research.google.com/drive/1kA_Md8ITx8F0KdRspmO7ltvaMAt8o6Q9#scrollTo=GSWiCxlJj72V)