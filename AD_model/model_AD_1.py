import torch
from pandarallel import pandarallel
import torch.nn as nn
from torch.nn import functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT
import numpy as np
from tqdm import tqdm

class AD(nn.Module):
    def __init__(self, num_entities, emb_dim ):
        super(AD, self).__init__()
        self.num_entities = num_entities
        self.emb = nn.Embedding(num_entities,emb_dim)
        self.mode = 'train'
        return


    def calc_score(self, x, neg_sample=False):
        x = self.emb(x)
        x = torch.sum(x, dim=1, keepdim=False)
        x = torch.norm(x, p=2, dim=-1)
        x = torch.pow(x,2)
        x = x.unsqueeze(-1)
        if neg_sample:
            x = torch.reciprocal(x)
        score = torch.tanh(x)
        return score

    def forward(self, x_pos, x_neg=None):
        if self.mode == 'train':
            scores_p = self.calc_score(x_pos)
            num_neg_samples = x_neg.shape[1]
            # Split negative samples
            list_x_n = torch.chunk(x_neg, chunks=num_neg_samples, dim=1)
            list_scores_n = []
            for x_n in list_x_n:
               
                x_n =  x_n.squeeze(1)
                _score = self.calc_score(x_n,True)
              
                list_scores_n.append(_score)
            scores_n = torch.cat(list_scores_n,dim=1)

            scores_n = torch.log(scores_n)
            scores_n = torch.mean(scores_n,dim=-1,keepdim=True)
            scores_p = torch.log(scores_p)
            sample_scores = scores_n + scores_p
            batch_score_mean = torch.mean(torch.squeeze(sample_scores))
            return batch_score_mean
        else:
            scores = self.calc_score(x_pos)
            return torch.squeeze(scores)

class AD_model_container():
    def __init__(self, entity_count, emb_dim):
        self.model = AD ( entity_count, emb_dim)
        return

    def train_model(self, train_x_pos, train_x_neg, batch_size = 512, epochs = 10, log_interval=100):
        self.model.mode='train'
        bs = batch_size
        opt = torch.optim.Adam( list(self.model.parameters()) )
        num_batches = train_x_pos.shape[0] // bs + 1
        idx = np.arange(train_x_pos.shape[0])
        loss_value_history = []
        for e in tqdm(range(epochs)):
            np.random.shuffle(idx)
            for b in range(num_batches):
                opt.zero_grad()
                b_idx = idx[b*bs:(b+1)*bs]
                x_p = LT(train_x_pos[b_idx])
                x_n = LT(train_x_neg[b_idx])
                
                loss = -self.model(x_p,x_n)
                loss.backward()
                opt.step()
                loss_value_history.extend(loss.data.numpy().to_list())
                if b % log_interval == 0 :
                    print('Epoch {}  batch {} Loss {:4f}'.format(e,b, loss.data.numpy()))
        try:
            import matplotlib.pyplot as plt
            y = loss_value_history
            x = np.arange(len(loss_value_history))
            plt.plot(x,y,'r')
            plt.ylabel('Loss')
            plt.xlabel('Batch')
            plt.show()
            plt.close()
        except:
            pass
        return

    def score_samples(self,x_test):
        bs = 507
        results = []
        num_batches = x_test.shape[0] // bs + 1
        idx = np.arange(x_test.shape[0])
        for b in range(num_batches):
            b_idx = idx[b * bs:(b + 1) * bs]
            if len(b_idx)==0: break
            x = LT(x_test[b_idx])
            score_values = self.model(x)
            vals = score_values.dat.numpy().tolist()
            results.extend(vals)
        return results

# data_p = np.array([[1,2,3],[9,7,3]])
# data_n = np.array([[[7,4,5],[4,3,9],[7,4,5]], [[7,6,5],[7,4,5],[4,8,1]]])
#
#
# obj = AD(10,4)
# p = LT(data_p)
# n = LT(data_n)
# obj(p,n)
# obj.mode= 'test'
# print(obj(p))



