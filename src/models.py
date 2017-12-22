import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class FakeNet(torch.nn.Module):
    def __init__(self):
        super(FakeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2444, 256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 4),
            nn.LeakyReLU()
            )
        # self.input = torch.nn.Linear(2444, 256)
        # self.hl1 = torch.nn.Linear(256, 64)
        # self.hl2 = torch.nn.Linear(64, 16)
        # self.last = torch.nn.Linear(16, 4)

    def forward(self, X):
        if isinstance(X, np.ndarray) or isinstance(X, list):
            X = torch.autograd.Variable(torch.FloatTensor(X))
        output = self.model(X)
        # inp_layer = self.input(X)
        # hidden1 = F.dropout(F.leaky_relu(self.hl1(inp_layer)), 0.3)
        # hidden2 = F.dropout(F.leaky_relu(self.hl2(hidden1)), 0.3)
        # output = F.leaky_relu(self.last(hidden2))
        return output

    def fit(self, X, y, wts=None):
        mapping = {'agree': 0, 'disagree':1, 'discuss':2, 'unrelated':3}
        revmapping = {v:k for k, v in mapping.items()}
        self.init_params()
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        if wts is not None and isinstance(wts, list):
            wts = torch.FloatTensor(wts)
        losses = []
        for epoch in tqdm(range(20)):
            bs = 200
            tloss = 0.0
            for i in range(0, len(y)-bs+1, bs):
                opt.zero_grad()
                pred = self.forward(X[i:i+bs]) #prediction on batch features
                yb = y[i:i+bs] # batch target
                if isinstance(yb, list):
                    yb = list(map(lambda x: mapping[x], yb)) # str labels to indices
                    yb = torch.autograd.Variable(torch.LongTensor(yb))
                if isinstance(yb, np.ndarray):
                    yb = torch.autograd.Variable(torch.LongTensor(yb))
                loss = F.cross_entropy(pred, yb, weight=wts)
                tloss += loss.data[0]
                loss.backward()
                opt.step()
            losses.append(tloss)
            #if epoch%10==0:
            #print(epoch, "::", tloss)
        return losses

    def predict(self, X):
        result = torch.max(F.log_softmax(self.forward(X)), 1)[1]
        return list(result.data.numpy())

    def init_params(self, method='normal'):
        if method == 'normal':
            init_scheme = nn.init.xavier_normal
        else:
            init_scheme = nn.init.xavier_uniform
        for i, param in enumerate(self.parameters()):
            if len(param.data.size()) < 2: 
                param.data = torch.ones(param.size()[0])/100
            else:
                param.data = init_scheme(param.data)