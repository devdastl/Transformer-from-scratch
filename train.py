import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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

class SimpleBigramModel(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embedding_size = embed_size
        self.embedding_matrix = torch.nn.Embedding(self.embedding_size, self.embedding_size)
        

    def forward(self, input, output=None):
        logits = self.embedding_matrix(input)  # 32x8 --> 32x8x65

        if output != None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), output.view(B*T))  #here BxT = 32*8 == 256
            return loss, logits
        else:
            return logits

    def infer(self, seed_int, size): #seed_int 1x1
        with torch.no_grad():
            for i in range(size):
                logits = self.forward(seed_int)  #first:iter 1x1x65, #second:iter 1x2x65, ....#size:iter 1xsizex65
                print(logits.shape)
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
            logits, loss = model_instance(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model_instance.train()
    return out

def main():

    #important hyperparameter:
    epoch = 100000
    device = 'cpu'
    batch_size = 32
    block_size = 8
    train_split = 0.8
    lr = 1e-5

    #initialize custom dataloader
    dataloader_instance = GptDataloader('dataset.txt',train_test_split=train_split)

    #initialize model instance
    model_instance = SimpleBigramModel(len(dataloader_instance.unique_char)).to(device)
    if os.path.exists("bigram_model.pth"):
        model_instance.load_state_dict(torch.load("bigram_model.pth"))

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

    input, output = dataloader_instance.get_batch(batch_size, block_size, 'train')
    loss, _ = model_instance(input, output)
    print(f"loss after {epoch} is -> {loss.item()}")
    torch.save(model_instance.state_dict(), "bigram_model.pth")
    estimate_loss(dataloader_instance, model_instance)


if __name__ == '__main__':
    main()


