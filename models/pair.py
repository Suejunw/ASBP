import sys
import math
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.functional import normalize, linear
from torch.nn import init
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class BertPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Pair(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 2)
        self.init_w = torch.nn.Parameter(torch.normal(0, 0.01, (2, 768)))
        self.anchor1 = torch.nn.Parameter(torch.tensor(0.3))
        self.anchor2 = torch.nn.Parameter(torch.tensor(0.3))
        # self.init_w1 = torch.nn.Parameter(torch.normal(0, 0.01, (1, 1068)))
        # self.init_w = torch.nn.Parameter(torch.Tensor(2, 768))
        # init.xavier_uniform_(self.init_w)

        self.pool = BertPooler()


    def forward(self, batch, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        output = self.sentence_encoder(batch)
        pooled_output = output[1]

        sequence_output = output[0]
        mean_output = torch.mean(sequence_output, dim=1)
        mean_output = self.pool(mean_output)
        pooled_output = self.drop(mean_output)

        # To draw a scatterplot, need to modify in dataloader.py

        # data = pooled_output.cpu().numpy()
        # label = np.tile(np.array([0,0,0,0,0,1,1,1,1,1]), 40)
        # tsne = TSNE(n_components=2, learning_rate=100, n_iter=1000).fit_transform(data)
        # plt.scatter(tsne[:, 0], tsne[:, 1], c=label)
        # figure = plt.gcf()
        # figure.savefig('gai.eps', format='eps',dpi =1500)

        norm_out = normalize(pooled_output, dim=-1)
        norm_w = normalize(self.init_w, dim=-1)

        sita1 = linear(norm_out, norm_w[0].unsqueeze(0))
        sita1.acos_()
        sita1 += 0.3
        logits1 = torch.mul(sita1.cos_(), 64)

        sita2 = linear(norm_out, norm_w[1].unsqueeze(0))
        sita2.acos_()
        sita2 += 0.3
        logits2 = torch.mul(sita2.cos_(), 64)

        logits = torch.cat((logits2,logits1), dim=-1)
        
        # logits = self.linear(pooled_output)

        logits = logits.view(-1, total_Q, N, K, 2)
        logits = logits.mean(3) # (-1, total_Q, N, 2)
        logits_na, _ = logits[:, :, :, 0].min(2, keepdim=True) # (-1, totalQ, 1)
        logits = logits[:, :, :, 1] # (-1, total_Q, N)
        logits = torch.cat([logits, logits_na], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred


