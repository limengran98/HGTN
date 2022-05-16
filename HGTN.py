
""" 
@project:HGTN
@author:mengranli 
@contact:
@website:
@file: HGTN.py 
@platform: 
@time: 2021/7/17 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#from layers import GraphAttentionLayer, SpGraphAttentionLayer
import argparse
import numpy as np
"""
type_hyperedge  不同种类超边数目
num_layers     超边矩阵相乘次数

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE(nn.Module):

    def __init__(self, n_input,n_enc,n_z ):
        super(AE, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc)

        self.z_layer = nn.Linear(n_enc, n_z)


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))

        z = self.z_layer(enc_h1)


        return enc_h1,z

class HGTN(nn.Module):
    def __init__(self, type_hyperedge, node_number, pretrain_path,num_channels, w_in, w_out, num_class,num_layers,norm,nhid, dropout, alpha, nheads,a,b):
        super(HGTN, self).__init__()
        self.type_hyperedge = type_hyperedge
        self.node_number =  node_number
        selfpretrain_path=pretrain_path, 
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.is_norm = norm
        self.nhid = nhid
        self.drpoput = dropout
        self.nheads = nheads
        self.alpha = alpha
        self.gat = GAT(self.w_in,self.nhid,self.num_class,self.drpoput ,self.alpha ,self.nheads)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=True))
            else:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.hgc1 = HGNN_conv(self.w_in,128)
        self.hgc2 = HGNN_conv(128,self.w_out)
        self.weighte1 = nn.Parameter(torch.Tensor(self.w_in,128))
        self.weighte2 = nn.Parameter(torch.Tensor(128,self.w_out))
        self.weightw = nn.Parameter(torch.Tensor(self.node_number,self.node_number))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.hgcnv_conv1 = nn.Linear(self.w_in, 128)
        self.hgcnv_conv2 = nn.Linear(128, self.w_out)
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.linear = nn.Linear( self.w_out* self.num_channels, self.num_class)
        self.mlp = MLP(128*2)
        self.class_layer = nn.Parameter(torch.Tensor(self.num_class, self.w_out))
        torch.nn.init.xavier_normal_(self.class_layer.data)
        self.reset_parameters()
        self.ae = AE(
            n_input = self.w_in,
            n_enc=128,
            n_z=self.w_out)
        self.ae.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
        self.a = a
        self.b = b
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weighte1)
        nn.init.xavier_uniform_(self.weighte2)
        nn.init.eye_(self.weightw)
        nn.init.zeros_(self.bias)

#    def hgcnv_conv(self,X,H):
#        support = torch.mm(X, self.weightv)
#        W= torch.mm(H,self.weightw)
#        return torch.mm(W,support)
    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
        
    def hgcne_conv1(self,X,H):
        support = torch.mm(X, self.weighte1)
        W = torch.mm(H.T,self.weightw)
        return torch.mm(W,support)
        
    def hgcne_conv2(self,X,H):
        support = torch.mm(X, self.weighte2)
        W = torch.mm(H.T,self.weightw)
        
        return torch.mm(W,support)
        #return X
    def feature_smoothing(self,adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat
    def forward(self, A, feature, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                
                H, W = self.layers[i](A)

            else:
                H, W = self.layers[i](A, H)
            Ws.append(W)
        #多通道
        for i in range(self.num_channels):
            if i==0:
                #
                X1,X11 = self.ae(feature)
                #X1 = F.relu(self.hgcnv_conv1(feature))
                X2 = F.relu(self.hgc1(feature,H[i]))
                #X11 = self.hgcnv_conv2(X1)

                m = self.mlp(torch.cat((X1,X2), 1) )
                m = F.normalize(m,p=2)
                m1 =  m[:,0].reshape(-1,1)
                m2 =  m[:,1].reshape(-1,1)

                X = m1*X1+m2*X2
                X = m1*X1+m2*X2

                X_ = self.hgc2(X,H[i])

            else:
                X1,X11 = self.ae(feature)
                #X1 = self.hgcnv_conv(feature)  #n,123
                X2 = F.relu(self.hgc1(feature,H[i])) #n,128
                

                m = self.mlp( torch.cat((X1,X2), 1) )
                m = F.normalize(m,p=2)
                m1 =  m[:,0].reshape(-1,1)
                m2 =  m[:,1].reshape(-1,1)

                X = m1*X1+m2*X2  #n,128
                X_tmp = F.relu(self.hgc2(X,H[i]))  #n,20
                #X = torch.cat((X1,X2), dim=1)
                #X_tmp = self.hgcn_conv(feature,H[i])
                X_ = torch.cat((X_,X_tmp), dim=1)  #n,20*c
       
        #X_ = self.hgcn_conv(X,H[0])
        #X_ = self.gat(X,H[0])    
        #X_ = F.dropout(X_, self.dropout,training=self.training)
        #X = F.softmax(self.linear1(X))  
        
        q = 1.0 / (1.0 + torch.sum(torch.pow(X11.unsqueeze(1) - self.class_layer, 2), 2) )
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        p = self.target_distribution(q)
        X_ = self.linear(X_) #n,3
        z = F.softmax(X_, dim=1)
        
        y = X_[target_x]
        #q= q[target_x]
        #p = p[target_x]
        #print(p.shape)
        
        #X_ = self.linear1(X_)
        #X_ = F.relu(X_)
        #y = self.linear2(X_[target_x])
        re_loss = self.loss(y, target)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(z.log(), p, reduction='batchmean')
        loss_smooth_feat = self.feature_smoothing(H[i], feature)
        loss =   re_loss  #+ self.a*kl_loss + self.b*ce_loss #+ 0.001*loss_smooth_feat
        #print('re_loss:{}'.format(re_loss))
        #print('kl_loss:{}'.format(kl_loss))
        #print('ce_loss:{}'.format(ce_loss))
        #print('loss_smooth_feat:{}'.format(loss_smooth_feat))
        return loss, y, Ws, H, X1,X2
        
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x        
        
class SHGTN(nn.Module):
    def __init__(self, type_hyperedge, node_number,pretrain_path, num_channels, w_in, w_out, num_class,num_layers,norm,nhid, dropout, alpha, nheads,a,b):
        super(SHGTN, self).__init__()
        self.type_hyperedge = type_hyperedge
        self.node_number =  node_number
        self.pretrain_path=pretrain_path
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.is_norm = norm
        self.nhid = nhid
        self.drpoput = dropout
        self.nheads = nheads
        self.alpha = alpha
        self.gat = GAT(self.w_in,self.nhid,self.num_class,self.drpoput ,self.alpha ,self.nheads)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=True))
            else:
                layers.append(HGTLayer(type_hyperedge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(self.w_in,self.w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.linear = nn.Linear( self.w_out* self.num_channels, self.num_class)
        self.mlp = MLP(self.w_out*3)
        self.dropout = dropout
        self.hgc1 = HGNN_conv(self.w_in,self.w_out)
        self.hgc2 = HGNN_conv(self.w_out* self.num_channels,self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def hgcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        return torch.mm(H,X)
        #return X

    def forward(self, A, feature, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)

            else:
                #H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        #多通道
        for i in range(self.num_channels):
            if i==0:
                X = F.relu(self.hgc1(feature,H[i]))
                X = F.dropout(X, self.dropout)
                #X = self.hgc2(x, G)


            else:
                X_tmp = F.relu(self.hgc1(feature,H[i]))
                X_tmp = F.dropout(X_tmp, self.dropout)
                #X_tmp = self.hgc2(x, G)
                X = torch.cat((X,X_tmp), dim=1)
                
        #X_ = self.linear(X)   
        #y = X_[target_x]
        
        X_ = self.hgc2(X,H[i])
        y = X_[target_x]
        
        
        #X_ = self.linear1(X_)
        #X_ = F.relu(X_)
        #y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws, H

        


class MLP(nn.Module):
    def __init__(self, n_mlp):
        super(MLP, self).__init__()
        self.wl = nn.Linear(n_mlp, 2)
    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        
        return weight_output

class HGTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(HGTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = HGNN_Conv(in_channels, out_channels)
            self.conv2 = HGNN_Conv(in_channels, out_channels)
        else:
            self.conv1 = HGNN_Conv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)

            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W


class HGNN_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(HGNN_Conv, self).__init__()

        self.weight = Parameter(torch.Tensor(out_channels,in_channels,1,1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.1)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, A):
        Q = torch.sum(A*F.softmax(self.weight, dim=1),dim=1)
        #print(F.softmax(self.weight, dim=1))
        return Q

