#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import AttentionalGNN
import neighbor_descriptor as nd

def MLP(in_channels:list, do_bn=True):
    n = len(in_channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(in_channels[i-1], in_channels[i], kernel_size=1, bias=True))
        if i < n-1:
            if do_bn:
                layers.append(nn.BatchNorm1d(in_channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def MLP2D(in_channels:list, do_bn=True):
    n = len(in_channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv2d(in_channels[i-1], in_channels[i], kernel_size=1, bias=True))
        if i < n-1:
            if do_bn:
                layers.append(nn.BatchNorm2d(in_channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder2D(nn.Module):
    def __init__(self, feature_dim, layers):
        super(KeypointEncoder2D,self).__init__()
        self.encoder = MLP2D([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, x):
        x = self.encoder(x.to(torch.float32))
        return x

class KeypointEncoder1D(nn.Module):
    def __init__(self, feature_dim, layers):
        super(KeypointEncoder1D,self).__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, x):
        x = self.encoder(x.to(torch.float32))
        return x

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  
    xx = torch.sum(x ** 2, dim=1, keepdim=True) 
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx,k

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx,k = knn(x, k=k)  
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  

    idx = idx + idx_base  

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  
    return feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args['k']
        output_channels = args['output_dim']
        emb_dims = 1024

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Conv1d(emb_dims,512,1)
        self.linear2 = nn.Conv1d(512, 256, 1)
        self.linear3 = nn.Conv1d(256, output_channels, 1)

    def forward(self, data_config):
        theta_distri = data_config['theta_dis']

        theta_dis = theta_distri.type(torch.FloatTensor).cuda()
        x = get_graph_feature(theta_dis.transpose(2,1).contiguous(), k=self.k)  
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0] 

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)  

        x = self.conv5(x)                     

        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        pred = self.linear3(x).squeeze(0).transpose(1,0).contiguous()
        return pred


class CrossScan(nn.Module):
    def  __init__(self, config):
        super(CrossScan, self).__init__()
        self.config = config

        self.kenc2D = KeypointEncoder2D(self.config['embed_dim'], self.config['keypoint_encoder'])
        self.kenc1D = KeypointEncoder1D(self.config['embed_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(self.config['embed_dim'], self.config['layer_names'])
    

    def forward(self, kpts0, kpts1, Nei_sle, kl):
        point1 = kpts0[0].detach().cpu().numpy()
        point2 = kpts1[0].detach().cpu().numpy()
        Nei_index = Nei_sle.detach().cpu().numpy()
        idx,NeiX,NeiY = nd.select_neighbor(point1,point2,Nei_index,kl)


        axisNeiX, axisNeiY = torch.tensor(NeiX).cuda(), torch.tensor(NeiY).cuda()
        Nei_gX = (axisNeiX[:, 1:] - axisNeiX[:, 0].unsqueeze(1))
        Nei_gY = (axisNeiY[:, 1:] - axisNeiY[:, 0].unsqueeze(1))
        Nei_ggX = torch.cat((Nei_gX, axisNeiX[:, 1:]), dim=-1)
        Nei_ggY = torch.cat((Nei_gY, axisNeiY[:, 1:]), dim=-1)
        NeiDesx = self.kenc1D(Nei_ggX.transpose(2, 1))
        NeiDesy = self.kenc1D(Nei_ggY.transpose(2, 1))
        Desx, Desy = self.gnn(NeiDesx, NeiDesy)  

        euclidean_distance = F.pairwise_distance(Desx, Desy, keepdim=True) 
        Dis1 = torch.min(euclidean_distance, dim=-1)[0]
        DisSum = Dis1
        return DisSum,idx



