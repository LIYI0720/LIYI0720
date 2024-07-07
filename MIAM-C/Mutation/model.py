import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import os, torchvision, torch
import torch.nn.functional as F
from cvt import CvT
from sklearn import svm

class AttentionSlide_Batch(nn.Module):
    def __init__(self):
        super(AttentionSlide_Batch, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        self.resnet = models.resnet34(pretrained=True)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L*self.K, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = x.view(x.shape[0]*x.shape[1], 3, x.shape[3], x.shape[4]) #squeeze batchsize for conv parallel.
        H = self.resnet(x1)
        A = self.attention(H)  # NxK
        # print(A.shape)
        A = A.view(x.shape[0], x.shape[1], -1)# recovery batchsize
        A = torch.transpose(A, 2, 1)  # KxN
        # print(A.shape)
        A = F.softmax(A, dim=2)  # softmax over N
        H = H.view(x.shape[0], x.shape[1], -1)
        M = torch.matmul(A, H)  # KxL attention to channel
        M = M.squeeze(dim=1)
        M = self.flatten(M)
        # print(M.shape)
        Y_prob = self.classifier(M)
        Y_prob = Y_prob.squeeze()
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A.squeeze()

class AttentionSlide_MultiBatch(nn.Module):
    def __init__(self):
        super(AttentionSlide_MultiBatch, self).__init__()
        self.L = 1000
        self.D = 128
        self.K = 1
        # self.resnet = models.resnet34(pretrained=False)
        self.resnet = models.resnet34(pretrained=True)
        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier1 = nn.Linear(self.L*self.K, 1)

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.classifier2 = nn.Linear(self.L*self.K, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = x.view(x.shape[0]*x.shape[1], 3, x.shape[3], x.shape[4]) #squeeze batchsize for conv parallel.
        
        H = self.resnet(x1)
        features = H
        A1 = self.attention1(H)  # NxK
        A2 = self.attention2(H)
        # print(A.shape)
        A1 = A1.view(x.shape[0], x.shape[1], -1)# recovery batchsize
        A1 = torch.transpose(A1, 2, 1)  # KxN
        A2 = A2.view(x.shape[0], x.shape[1], -1)# recovery batchsize
        A2 = torch.transpose(A2, 2, 1)  # KxN
        # print(A.shape)
        A1 = F.softmax(A1, dim=2)  # softmax over N
        A2 = F.softmax(A2, dim=2)
        H = H.view(x.shape[0], x.shape[1], -1)
        M1 = torch.matmul(A1, H)  # KxL attention to channel
        M1 = M1.squeeze(dim=1)
        M1 = self.flatten(M1)
        M2 = torch.matmul(A2, H)  # KxL attention to channel
        M2 = M2.squeeze(dim=1)
        M2 = self.flatten(M2)
        # print(M.shape)
        wild_prob1 = self.classifier1(M1)
        wild_prob1 = wild_prob1.squeeze(dim = 1)
        wild_prob2 = self.classifier2(M2)
        wild_prob2 = wild_prob2.squeeze(dim = 1)
        
        Y_prob = torch.stack((wild_prob1,wild_prob2),dim=1)
        Y_prob_c = F.softmax(Y_prob,dim=1)

        #返回最大值的索引
        #第一个tensor是每行的最大值value
        #第二个tensor是每行最大值的索引index
        _, Y_hat = torch.max(Y_prob_c, 1)

        A = torch.stack((A1,A2),dim=1).squeeze()

        return Y_prob_c, Y_hat
        # return features




class my_resnet_base(nn.Module):
    def __init__(self) :
        super(my_resnet_base,self).__init__()
        self.L = 1000
        self.D = 256
        self.class_num= 2

        self.resnet = models.resnet34(pretrained=True)


        self.fc3  = nn.Sequential(
            nn.Linear(self.L,self.D),
            nn.Tanh(),
            nn.Linear(self.D,self.class_num),

        ) 

    def forward(self,x):
        x = x.view(x.shape[0],3,256,256)
        out = self.resnet(x)
        out = self.fc3(out)
        # out = out.view(x.shape[0],-1)
        Y_prob = F.softmax(out,dim=1)
        Y_prob = Y_prob.squeeze()
        _,Y_hat = torch.max(Y_prob,1)
        return Y_prob,Y_hat