import torch
from pandarallel import pandarallel
import torch.nn as nn
from torch.nn import functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import time
from time import time
import os
from pathlib import Path

class AD(nn.Module):
    def __init__(self, num_entities, emb_dim , device):
        super(AD, self).__init__()
        self.device = device
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
            scores_n = torch.sum(scores_n,dim=-1,keepdim=True)
            scores_p = torch.log(scores_p)
            sample_scores = scores_n + scores_p
            batch_score_mean = torch.mean(torch.squeeze(sample_scores))
            return batch_score_mean
        else:
            scores = self.calc_score(x_pos)
            return torch.squeeze(scores)

class AD_model_container():

    def __init__(self, entity_count, emb_dim, device, lr = 0.0005):
        self.model = AD ( entity_count, emb_dim, device)
        self.device = device
        
        self.model.to(self.device)
        self.lr = lr
        self.entity_count = entity_count
        self.emb_dim = emb_dim
        self.signature = 'model_{}_{}'.format(emb_dim,int(time()))
        self.save_path = None
        self.epoch_meanLoss_history = []
        return

    def train_model(self, train_x_pos, train_x_neg, batch_size = 512, epochs = 10, log_interval=100, tol = 0.025):
        self.model.mode = 'train'
        bs = batch_size
        opt = torch.optim.Adam( list(self.model.parameters()), lr = self.lr)
        num_batches = train_x_pos.shape[0] // bs + 1
        idx = np.arange(train_x_pos.shape[0])
        loss_value_history = []
        clip_value = 5
        loss_history = []


        for e in tqdm(range(epochs)):
            np.random.shuffle(idx)
            epoch_loss =[]
            pbar = tqdm(range(num_batches))
            for b in pbar:
                opt.zero_grad()
                b_idx = idx[b*bs:(b+1)*bs]
                x_p = LT(train_x_pos[b_idx]).to(self.device)
                x_n = LT(train_x_neg[b_idx]).to(self.device)
                
                loss = -self.model(x_p, x_n)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                opt.step()
                loss_value_history.append(loss.cpu().data.numpy().tolist())
                tqdm._instances.clear()
                pbar.set_postfix({'Batch ': b + 1})
                if b % log_interval == 0 :
                    print('Epoch {}  batch {} Loss {:4f}'.format(e, b, loss.cpu().data.numpy()))
                epoch_loss.append(loss.cpu().data.numpy())

            self.epoch_meanLoss_history.append(np.mean(epoch_loss))
            loss_history.extend(epoch_loss)
            print('Mean epoch loss {:.4f}'.format(np.mean(epoch_loss)))

            if len(self.epoch_meanLoss_history) > 10:
                delta1 = abs(self.epoch_meanLoss_history[-2] - self.epoch_meanLoss_history[-1])
                delta2 = abs(self.epoch_meanLoss_history[-3] - self.epoch_meanLoss_history[-2])

                if delta2 <= tol and delta1 <= tol:
                    print('Stopping!')
                    break


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
        self.model.mode = 'test'
        return

    def score_samples(self, x_test):
        bs = 507
        results = []
       
        num_batches = x_test.shape[0] // bs + 1
        idx = np.arange(x_test.shape[0])
        for b in range(num_batches):
            b_idx = idx[b * bs:(b + 1) * bs]
            if len(b_idx)==0 : 
                break
            
            x = LT(x_test[b_idx]).to(self.device)
           
            score_values = self.model(x)
            vals = score_values.cpu().data.numpy().tolist()
            results.extend(vals)
        return results

    def save_model(self, loc=None):
        if loc is None:
            loc = './saved_models'
        path_obj = Path(loc)
        path_obj.mkdir(parents=True, exist_ok=True)
        loc = os.path.join(loc, self.signature )
        self.save_path = loc
        torch.save(self.model.state_dict(), loc)


    def load_model(self, path = None):
        if self.save_path is None and path is None:
            print('Error . Null path given to load model ')
            return None
        print('Device', self.device)
        if path is None:
            path = self.save_path
        self.model = AD( emb_dim=self.emb_dim, num_entities=self.entity_count,device=self.device)
        
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        self.model.mode = 'test'
        return

    def predict(self,x_test):
        return self.score_samples(x_test)

    


