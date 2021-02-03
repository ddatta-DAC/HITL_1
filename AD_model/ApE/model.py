import torch
import os
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch import FloatTensor as FT



class APE(nn.Module):
    def __init__(self, num_domains, emb_dim, domain_dims):
        super(APE, self).__init__()

        self.num_domains = num_domains
        self.emb_dim = emb_dim

        self.num_entities = sum(domain_dims.values())
        print( self.num_entities)
        self.emb_layer = nn.Embedding(
            num_embeddings=self.num_entities,
            embedding_dim=emb_dim
        )
        print('hello')
        print(self.emb_layer)
        self.c = nn.Parameter(torch.from_numpy(np.random.uniform()))
        print(self.c)
        k = 0

        self.pair_W = nn.ModuleDict({})
        for i in range(num_domains):
            for j in range(i+1,num_domains):
                w_k = nn.Parameter(data= torch.from_numpy(np.random.random(1)))
                self.pair_W[k] = w_k
                k+=1
        return

    def obtain_score(self,x):
        x = self.emb_layer(x)
        # Split along axis = 1
        x1 = [_.squeeze(1) for _ in  torch.chunk( x, self.num_domains, dim=1)]
        k = 0
        _score = []
        for i in range(self.num_domains):
            for j in range(i+1,self.num_domains):
                r = torch.sum(torch.mul(x1[i],x1[j]), dim=-1, keepdims=True)
                r = r * torch.exp(self.pair_W[k])
                _score.append(r)
        _score = torch.cat(_score, dim=-1)
        score = torch.exp(torch.sum(_score,dim=-1) + self.c)
        self.mode = 'train'
        return score

    # Expect entitty ids to be serialized
    def forward(self, pos_x, neg_x = None):
        pos_score = self.obtain_score(pos_x)
        if self.mode =='test':
            return pos_score
        num_neg_samples = neg_x.shape[1]
        neg_x_split = [ _.unsqueeze(1) for _ in torch.chunk(neg_x,num_neg_samples,dim=1)]
        neg_score = []
        for nsample_i in neg_x_split:
            s = self.obtain_score(nsample_i)
            neg_score.append(s)
        neg_score = torch.cat(neg_score, dim=-1)
        return pos_score,neg_score

    def score_records(self, x):
        self.mode= 'test'
        self.eval()
        return self.obtain_score(x)

class APE_container:
    def __init__(self, model_obj, batch_size= 128, LR=0.0001 ):

        self.model_obj = model_obj
        self.optimizer = torch.optim.Adam(self.model_obj.parameters, lr=LR)
        self.batch_size = batch_size
        return

    def train_model(self, pos_x, neg_x, num_epochs =50):
        self.model_obj.train()
        self.model_obj.mode ='train'
        clip_value = 5
        bs = self.batch_size
        idx = np.arange(pos_x.shape[0])
        num_batches = pos_x.shape[0]//bs + 1
        for e in tqdm(num_epochs):
            np.random.shuffle(idx)
            for b in tqdm(range(num_batches)):
                self.optimizer.zero_grad()
                _idx = bs[b*bs : (b+1)*bs]
                _x_pos = pos_x[_idx]
                _x_neg = neg_x[_idx]
                _x_pos = torch.FT(_x_pos)
                _x_neg = torch.FT(_x_neg)
                pos_score, neg_score = self.model_obj(_x_pos, _x_neg)
                # Calculate loss
                term1 = torch.log(torch.sigmoid(torch.log(pos_score)))
                term2 = torch.log(torch.sigmoid(-torch.log(neg_score)))
                print(term2.shape)
                term2 = torch.sum(term2, dim=-1)
                print(term2.shape)
                loss = term1 + term2
                loss = torch.mean(loss, dim=0)
                loss = -loss
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model_obj.parameters(), clip_value)
                self.optimizer.step()

        self.model_obj.mode='test'
        return

num_domains = 5
emb_dim= 16
domain_dims = { 'a' : 12, 'b': 15, 'c':10, 'd':8 , 'e': 5}
model_obj = APE(num_domains, emb_dim, domain_dims)

# container = APE_container(model_obj, batch_size= 16)
x1 = np.random.randint(0,50,[100,5])
