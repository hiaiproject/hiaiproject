# _*_coding:utf-8_*_#

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz, out_sz, layers=[512, 256, 128, 32]):
        super().__init__()
        # linear regression & drop out
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = torch.nn.BatchNorm1d(512)


        self.fc2 = nn.Linear(layers[0], layers[1])
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(layers[1], layers[2])
        self.dropout3 = nn.Dropout(0.3)
        self.bn3 = torch.nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(layers[2], layers[3])
        self.dropout4 = nn.Dropout(0.3)
        self.bn4 = torch.nn.BatchNorm1d(32)

        self.fc5= nn.Linear(layers[3], out_sz)
        self.bn5 = torch.nn.BatchNorm1d(out_sz)



    def forward(self, X):
        # ReLU로 Forward propogation 진행
        X = self.fc1(X)
        X = self.bn1(X)
        X = F.relu(X)
        X = self.dropout1(X)

        X = self.fc2(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = self.dropout2(X)

        X = self.fc3(X)
        X = self.bn3(X)
        X = F.relu(X)
        X = self.dropout3(X)

        X = self.fc4(X)
        X = self.bn4(X)
        X = F.relu(X)
        X = self.dropout4(X)

        # 마지막 layer는 softmax
        X = self.fc5(X)
        X = self.bn5(X)


        return F.log_softmax(X, dim=1)
