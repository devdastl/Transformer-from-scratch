#This is version 4 of the code with multi head attention. Following are new changes:
##usecase change:
# Now we will be training multi headed attention based GPT on our tiny shakespeare data

### Architecture change: 
#1. Added Multiheaded attention class which will create multiple attention head by dividing our embedding size.
#2. Added a Feed Forward layer after multiheaded attention block
#3. Added Block class which will repeat (multiheaded attention + feed forward layer)
#4. Added optimizations for deep neural network like residual connection and feature normalization(LayerNorm).
#5. Added dropouts for regularization.

### Bug fixes:
#1. Fixed issue with evaluation code. Now it is working as expected, other version of code does not have fix this issue.


import torch
import torch.nn as nn
import torch.nn.functional as F
import os

DEVICE = 'cpu'
DROPOUT = 0.2

#Create a class which will handle dataset, conversion and batches.
class GptDataloader():
    def __init__(self, text_file, train_test_split=0.8):
        with open(text_file, 'r') as f:   #read the data
            all_text = f.read()

        self.dataset_length = len(all_text)
        self.unique_char = sorted(list(set(all_text)))  #Extract all unique characters
        self.char_to_int = {char:i for i, char in enumerate(self.unique_char)}   #character to int mapping
        self.int_to_char = {i:char for i,char in enumerate(self.unique_char)}    #integer to char mapping
        encode_chars = self.encode(all_text)

        train_size = int(self.dataset_length * train_test_split)
        self.train_set = encode_chars[0:train_size]
        self.test_set = encode_chars[train_size:]    
        
    def get_batch(self, batch_size, block_size, split_type='train'):
        data_set = self.train_set if split_type == 'train' else self.test_set
        random_index_batch = torch.randint(len(data_set) - block_size, (batch_size,))
        x = torch.stack([data_set[index : index + block_size] for index in random_index_batch])
        y = torch.stack([data_set[index+1 : index+block_size+1] for index in random_index_batch])
        return x,y

    def encode(self, list_char):
        return torch.tensor([self.char_to_int[char] for char in list_char])
        
    def decode(self, int_list):
        return "".join([self.int_to_char[i] for i in int_list])

#Create single head attention module
class SingleAttentionHead(nn.Module):
    def __init__(self, block_size, embed_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.embed_size = embed_size

        self.key_matrix = nn.Linear(embed_size, head_size, bias=False)
        self.query_matrix = nn.Linear(embed_size, head_size, bias=False)
        self.value_matrix = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        #we define a variable which is not parameter as "register_buffer" with name 'tril'
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, input): #input --> BxTxembed_size
        batch, sequence, channel = input.shape
        key, query, value = self.key_matrix(input), self.query_matrix(input), self.value_matrix(input) #BxTxembed_size --> BxTxhead_size
        attention_weight = (query @ key.transpose(-2,-1)) * (channel ** -0.5) #BxTxembed @BxembedxT --> BxTxT #normalize to have 1 variance
        attention_weight = attention_weight.masked_fill(self.tril[:sequence,:sequence]==0, float('-inf'))
        attention_weight = F.softmax(attention_weight, dim=-1)
        attention_weight = self.dropout(attention_weight)
        out = attention_weight @ value

        return out

#multiple attention head running in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, embed_size, head_size, num_head):
        super().__init__()
        self.head_size = head_size
        self.num_head = num_head
        self.attention_head_list = nn.ModuleList([SingleAttentionHead(block_size, embed_size, head_size) for _ in range(num_head)])
        self.residual_projection = nn.Linear(head_size*num_head, embed_size)  
        self.dropout = nn.Dropout(DROPOUT)
        

    def forward(self, input):
        output = torch.cat([attention_instance(input) for attention_instance in self.attention_head_list], dim=-1)
        output = self.dropout(self.residual_projection(output))  #project output of multiheaded attention to embed_size so that it can be added with inputs
        return output

#Feed forward layer, it will be after multiheaded attention. It is let token do some computation after 
#passing through multiheaded attention whre they do communication with each other.
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.feedword_layer = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),  # here (embed_size, embed_size) can be (embed_size, any_number), multiply by 4 based on transformer paper
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),  #Residual projection: to project output of feedforward back to x.
            nn.Dropout(DROPOUT),
        )

    def forward(self, input):
        return self.feedword_layer(input)

#Transformer block which will contain multiheaded attention and feedforward layer.
class TransformerBlock(nn.Module):
    def __init__(self, block_size, embed_size, num_head):
        super().__init__()
        self.head_size = embed_size // num_head
        self.layer_norm1 = nn.LayerNorm(embed_size)  #layer norm to apply before going to self attention head
        self.layer_norm2 = nn.LayerNorm(embed_size)  #layer norm to apply before going to feedforward head
        self.multi_head_attention = MultiHeadAttention(block_size, embed_size, self.head_size, num_head)
        self.feedforward_layer = FeedForward(embed_size)

    def forward(self, input):
        input = input + self.multi_head_attention(self.layer_norm1(input))  # (input + atten_head) is a skip connection
        output = input + self.feedforward_layer(self.layer_norm2(input))    # similarly here as well, we have skip connection
        return output

class GPTModel(nn.Module):
    def __init__(self, embed_size, vocab_size, block_size,num_head, num_layer):
        super().__init__()
        self.embedding_size = embed_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_matrix = torch.nn.Embedding(self.vocab_size, self.embedding_size) #--> 32x8xembedding_size
        self.poisiton_embedding_matrix = torch.nn.Embedding(self.block_size, self.embedding_size)
        self.transformer_block = nn.Sequential(*[TransformerBlock(self.block_size, embed_size, num_head) for _ in range(num_layer)])
        self.layernorm = nn.LayerNorm(embed_size)
        self.linear_layer_Head = torch.nn.Linear(self.embedding_size, self.vocab_size) #--> 32x8xvocab_size

    def forward(self, input, output=None):
        Batch_size, sequence_size = input.shape
        token_embeddings = self.embedding_matrix(input)  # 32x8 --> 32x8xembed_size
        position_embedding = self.poisiton_embedding_matrix(torch.arange(sequence_size, device=DEVICE)) # 8xembed_size
        token_embeddings = token_embeddings + position_embedding #32x8x32 + 8x32 --> 32x8x32
        attn_output = self.layernorm(self.transformer_block(token_embeddings)) #32x8x32 --> 32x8x32
        
        logits = self.linear_layer_Head(attn_output) #32x8x32 --> 32x8x65

        if output != None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), output.view(B*T))  #here BxT = 32*8 == 256
            return loss, logits
        else:
            return logits

    def infer(self, seed_int, size, block_size): #seed_int 1x1
        with torch.no_grad():
            for i in range(size):
                indx_cond = seed_int[:,-block_size:]
                logits = self.forward(indx_cond)  #first:iter 1x1x65, #second:iter 1x2x65, ....#size:iter 1xsizex65
                proab_distribution = F.softmax(logits[:,-1,:], dim=-1) #calculating distribution across last char "-1" and last dim
                indx = torch.multinomial(proab_distribution, num_samples=1) 
                seed_int = torch.cat((seed_int, indx), dim=1)  #1x2
            return seed_int[0]

def test(dataloader_instance, model_instance):
    print(len(dataloader_instance.unique_char))
    print(dataloader_instance.encode('Devesh'))
    print(dataloader_instance.decode([16, 43, 60, 43, 57, 46]))
    print(len(dataloader_instance.train_set) + len(dataloader_instance.test_set))
    train_set, ground_set = dataloader_instance.get_batch(32,8, 'train')
    seed_int = torch.tensor([[0]])
    print(seed_int.shape)
    result = model_instance.infer(seed_int,100).tolist()
    print(result)
    print(dataloader_instance.decode(result))

@torch.no_grad()
def estimate_loss(dataloader_instance, model_instance):
    out = {}
    model_instance.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = dataloader_instance.get_batch(32,8, split)
            loss, logits = model_instance(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_instance.train()
    return out

def main():

    #important hyperparameter:
    epoch = 10000
    device = DEVICE
    batch_size = 32
    block_size = 8
    embedding_size = 32
    num_head = 4
    num_layer = 4
    train_split = 0.8
    lr = 1e-4

    #initialize custom dataloader
    dataloader_instance = GptDataloader('dataset.txt',train_test_split=train_split)

    #initialize model instance
    vocab_size = len(dataloader_instance.unique_char)
    model_instance = GPTModel(embedding_size, vocab_size, block_size, num_head, num_layer).to(device)
    model_instance.to(device)
    if os.path.exists("mutlihead_transformer_model.pth"):
        model_instance.load_state_dict(torch.load("mutlihead_transformer_model.pth"))

    #print total number of trainable parameters in model
    total_parameters = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    print(f"total trainable parameters are: {total_parameters}")

    #initialize optimizer to perform gradient decent.
    optimizer = torch.optim.AdamW(model_instance.parameters(), lr=lr)

    for i in range(epoch):
        input, output = dataloader_instance.get_batch(batch_size, block_size, split_type='train')
        input, output = input.to(device), output.to(device)
        loss, logits = model_instance(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")

    
    torch.save(model_instance.state_dict(), "mutlihead_transformer_model.pth")
    eval_loss = estimate_loss(dataloader_instance, model_instance)
    print(f"---- Total epochs: {epoch}, train_loss: {eval_loss['train']}, eval_loss: {eval_loss['val']} ----")

    #test inference
    seed_index = torch.tensor([[0]])
    print(seed_index.shape)
    print(dataloader_instance.decode((model_instance.infer(seed_index, size=200, block_size=block_size)).tolist()))

if __name__ == '__main__':
    main()


