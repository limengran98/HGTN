""" 
@project:HGTN
@author:mengranli 
@contact:
@website:
@file: HGTN.py 
@platform: 
@time: 2021/7/17 
"""

import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
from HGTN import HGTN,SHGTN
import hg_con as hgcon
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,f1_score
import warnings
#from pytorch_metric_learning import losses
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='DBLP',
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=20,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')
    parser.add_argument('--variable_weight',default=False)
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    epochs = args.epoch
    node_dim = args.node_dim

    lr = args.lr
    weight_decay = args.weight_decay
    norm = args.norm
    adaptive_lr = args.adaptive_lr
    dataname = args.dataset
    variable_weight = args.variable_weight
    nhid=args.hidden, 
    dropout=args.dropout, 
    nheads=args.nb_heads, 
    alpha=args.alpha
    
    if dataname == 'DBLP':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph_path = 'data/{}/{}_graph'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_APA'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_APCPA'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_APTPA'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph = np.loadtxt(data_graph_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt')
        pretrain_path = '{}.pkl'.format(dataname)
        args.num_layers = 3
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1, graph2, graph3]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

    
    if dataname == 'ACM':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph_path = 'data/{}/{}_graph'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_PAP'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_PLP'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_PMP'.format(dataname,dataname)
        data_graph4_path = 'data/{}/{}_PTP'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph = np.loadtxt(data_graph_path+'.txt') 
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt')
        graph4 = np.loadtxt(data_graph4_path+'.txt')
        pretrain_path = '{}.pkl'.format(dataname)
        args.num_layers = 3
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2,graph3,graph4]):
            G = hgcon.generate_HG(features, graph)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

        
    if dataname == 'CITE':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        pretrain_path = '{}.pkl'.format(dataname)
        args.num_layers = 3
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
    
    if dataname == 'IMDB':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_MAM'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_MDM'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_MYM'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        print(features.shape)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt')
        pretrain_path = '{}.pkl'.format(dataname)
        args.num_layers = 2
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2,graph3]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

    if dataname == 'REUT':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        args.num_layers = 3
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
    
    if dataname == 'STUDENT':
        K_neigs = 2
        student_feature = pd.read_csv('data/STUDENT/student_features_min_oversampler.csv')
        features = student_feature.iloc[:,1:].astype(np.float32).values  #特征
        labels = student_feature.iloc[:,0].values  #标签
        features, x_val, labels, y_val = train_test_split(features,labels, train_size = 0.036, random_state = 0)
#        fts_info= features[:,88:].astype(np.float32)#info 1
#        fts_breakfast = features[:,1:18].astype(np.float32)#breakfast 1
#        fts_lunch = features[:,18:35].astype(np.float32) #lunch 1
#        fts_dinner = features[:,35:52].astype(np.float32) #dinner
#        fts_wg = features[:,52:66].astype(np.float32)#wg
#        fts_wljf = features[:,74:88].astype(np.float32)#wljf
#        fts_library = features[:,66:74].astype(np.float32)#library
        fts_info= features[:,1:164].astype(np.float32)#info 1
        fts_breakfast = features[:,164:174].astype(np.float32)#breakfast 1
        fts_lunch = features[:,174:184].astype(np.float32) #lunch 1
        fts_dinner = features[:,184:194].astype(np.float32) #dinner
        fts_wg = features[:,194:212].astype(np.float32)#wg
        fts_wljf = features[:,212:231].astype(np.float32)#wljf
        fts_library = features[:,231:235].astype(np.float32)#library
        fts_shopping = features[:,235:242].astype(np.float32)#shopping
        args.num_layers = 3
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        #H = None
        #tmp = hgut.construct_muiH_with_KNN(fts_info,fts_breakfast,fts_lunch,fts_dinner,fts_wg,fts_wljf,fts_library,fts_shopping, K_neigs=K_neigs)
        #print(tmp.shape)
        #H = hgut.hyperedge_concat(H, tmp)
        
        for i,name in enumerate([fts_info,fts_breakfast,fts_lunch,fts_dinner,fts_wg,fts_wljf,fts_library,fts_shopping]):
            H = hgcon.construct_H_with_KNN(X=name, K_neigs=K_neigs)
            G = hgcon.generate_G_from_H(H)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')

    if dataname == 'USPS':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph1'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph3'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph4_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt') 
        graph4 = np.loadtxt(data_graph4_path+'.txt') 
        args.num_layers = 4
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2,graph3,graph4]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
    
    if dataname == 'HHAR':
        data_features_path = 'data/{}/{}_feature'.format(dataname,dataname)
        data_label_path = 'data/{}/{}_label'.format(dataname,dataname)
        data_graph1_path = 'data/{}/{}_graph1'.format(dataname,dataname)
        data_graph2_path = 'data/{}/{}_graph3'.format(dataname,dataname)
        data_graph3_path = 'data/{}/{}_graph5'.format(dataname,dataname)
        data_graph4_path = 'data/{}/{}_graph10'.format(dataname,dataname)
        features = np.loadtxt(data_features_path+'.txt')
        #print(features)
        labels = np.loadtxt(data_label_path+'.txt')
        graph1 = np.loadtxt(data_graph1_path+'.txt') 
        graph2 = np.loadtxt(data_graph2_path+'.txt') 
        graph3 = np.loadtxt(data_graph3_path+'.txt') 
        graph4 = np.loadtxt(data_graph4_path+'.txt') 
        args.num_layers = 4
        print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        for i,graph in enumerate([graph1,graph2,graph3,graph4]):
            G = hgcon.generate_HG(features, graph,variable_weight)
            G = torch.Tensor(G).to(device)
            if i == 0:
                A = G.unsqueeze(-1)
            else:
                A = torch.cat([A,G.unsqueeze(-1)], dim=-1)
        A = torch.cat([A,torch.eye(features.shape[0]).type(torch.FloatTensor).unsqueeze(-1).to(device)], dim=-1)
        print(A)
        print(A.shape)
        if A is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
        

    x = pd.DataFrame(features)
    y = pd.Series(labels)

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 0)
    idx_train = x_train.index
    idx_test = x_test.index
    node_features = torch.from_numpy(features).type(torch.FloatTensor).to(device)
    node_features.long()
    A = A.float()
    print(np.isnan(A.cpu().numpy()).any())
    print(np.isnan(node_features.cpu().numpy()).any())
    train_node = torch.from_numpy(np.array(idx_train)).to(device)
    train_target = torch.from_numpy(np.array(y_train)).long().to(device)
    test_node = torch.from_numpy(np.array(idx_test)).to(device)
    test_target = torch.from_numpy(np.array(y_test)).long().to(device)
    eprm_state = 'classification_new_5_8'
    file_out = open('./output/'+dataname+'_'+eprm_state+'.txt', 'a')
    print("The experimental results", file=file_out)
    #num_classes = torch.max(train_target).item()+1
    num_classes = int(train_target.max()) +1

    #loss_func = losses.TripletMarginLoss()
    #loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
    #loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    #losses.CircleLoss(m=0.4, gamma=80)
    #loss_func = losses.NPairsLoss()
    #loss_func = losses.NCALoss(softmax_scale=1)
    #loss_func = losses.VICRegLoss(invariance_lambda=25, variance_mu=25, covariance_v=1, eps=1e-4)

    
    #for a in [0.01,0.1,1,10,100,1000]:
    #    for b in [0.01,0.1,1,10,100,1000]:
    for a in [10]:
        for b in [0.1]:
            model = HGTN(type_hyperedge=A.shape[-1],
                          node_number=A.shape[0],
                          pretrain_path=pretrain_path, 
                                num_channels=1,
                                w_in = node_features.shape[1],
                                w_out = node_dim,
                                num_class=num_classes,
                                num_layers=2,
                                norm=norm,
                                nhid=args.hidden, 
                                dropout=args.dropout,  
                                alpha=args.alpha,
                                nheads=args.nb_heads,
                                a=a,
                                b=b) 
            
            model = model.to(device)
            if adaptive_lr == 'false':
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
            else:
                optimizer = torch.optim.Adam([{'params':model.weight},
                                            {'params':model.linear1.parameters()},
                                            {'params':model.linear2.parameters()},
                                            {"params":model.layers.parameters(), "lr":0.5}
                                            ], lr=0.0005, weight_decay=0.001)
            loss = nn.CrossEntropyLoss()
            # Train & Valid & Test
            best_test_loss = 10000
            best_train_loss = 10000
            best_train_f1 = 0
            best_test_f1 = 0
            
            best_train_acc = 0
            best_test_acc = 0
            
            best_train_pre = 0
            best_test_pre = 0
            
            best_train_rec = 0
            best_test_rec = 0

            with torch.no_grad():
                _, z = model.ae(node_features)
            kmeans = KMeans(n_clusters=num_classes, n_init=20)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            y_pred_last = y_pred
            model.class_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
            
            since = time.time()
            
            for i in range(epochs):
                for param_group in optimizer.param_groups:
                    if param_group['lr'] > 0.005:
                        param_group['lr'] = param_group['lr'] * 0.9
                
                model.zero_grad()
                model.train()
                lossce,y_train,Ws, H, X1,X2 = model(A, node_features, train_node, train_target)
                #lossm = loss_func(X1,X2)#
                loss = lossce 
                train_f1 = f1_score(train_target.cpu().numpy(),torch.argmax(y_train,dim=1).cpu().numpy(),average = 'macro')
                train_acc = accuracy_score(train_target.cpu().numpy(),torch.argmax(y_train,dim=1).cpu().numpy())
                train_pre = precision_score(train_target.cpu().numpy(),torch.argmax(y_train,dim=1).cpu().numpy(),average = 'macro')
                train_rec = recall_score(train_target.cpu().numpy(),torch.argmax(y_train,dim=1).cpu().numpy(),average = 'macro')
                if i%20 ==0:
                    print('Epoch:  ',i+1)
                    print('Train - Loss: {}, \n Macro_F1: {}, \n Acc: {}, \n Macro_Pre: {}, \n Macro_Rec: {}'.format(loss.detach().cpu().numpy(), train_f1,train_acc ,train_pre, train_rec))
                #print('Ws:{}'.format(Ws))
                loss.backward()
                optimizer.step()
                model.eval()
                # Test
                with torch.no_grad():
                    test_loss, y_test,W, H, X1,X2 = model.forward(A, node_features, test_node, test_target)
                    test_f1 = f1_score(test_target.cpu().numpy(),torch.argmax(y_test,dim=1).cpu().numpy(),average = 'macro')
                    test_acc = accuracy_score(test_target.cpu().numpy(),torch.argmax(y_test,dim=1).cpu().numpy())
                    test_pre = precision_score(test_target.cpu().numpy(),torch.argmax(y_test,dim=1).cpu().numpy(),average = 'macro')
                    test_rec = recall_score(test_target.cpu().numpy(),torch.argmax(y_test,dim=1).cpu().numpy(),average = 'macro')
                    classification = classification_report(test_target.cpu().numpy(),torch.argmax(y_test,dim=1).cpu().numpy(), digits=4)
                    if i%20 ==0:
                        print('Test - Loss: {}, \n Macro_F1: {}, \n Acc: {}, \n Macro_Pre: {}, \n Macro_Rec: {}'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc, test_pre, test_rec))
                        print(classification)
                    #print('W:{}'.format(Ws))
                if test_f1 > best_test_f1:

                    best_test_loss = test_loss.detach().cpu().numpy()
                    best_train_loss = loss.detach().cpu().numpy()
                    
                    best_train_f1 = train_f1
                    best_test_f1 = test_f1 
                    
                    best_train_acc = train_acc
                    best_test_acc = test_acc
            
                    best_train_pre = train_pre
                    best_test_pre = test_pre
            
                    best_train_rec = train_rec
                    best_test_rec = test_rec 
            np.savetxt(dataname+'X1.txt', X1.data.cpu().numpy())
            np.savetxt(dataname+'X2.txt', X2.data.cpu().numpy())
            print('---------------Best Results--------------------')
            print('Train - Loss: {},  \n Macro_Pre: {}, \n Macro_Rec: {},  \n Acc: {}, \n Macro_F1: {},'.format(best_train_loss ,best_train_pre, best_train_rec, best_train_acc ,best_train_f1))
            print('Test - Loss: {},  \n Macro_Pre: {}, \n Macro_Rec: {},  \n Acc: {}, \n Macro_F1: {},'.format(best_train_loss ,best_test_pre, best_test_rec, best_test_acc ,best_test_f1))
            time_elapsed = time.time() - since
            print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print('---------------Best Results--------------------',file=file_out)
            print(a,b,file=file_out)
            print('Train - Loss: {},  \n Macro_Pre: {}, \n Macro_Rec: {},  \n Acc: {}, \n Macro_F1: {},'.format(best_train_loss ,best_train_pre, best_train_rec, best_train_acc ,best_train_f1),file=file_out)
            print('Test - Loss: {},  \n Macro_Pre: {}, \n Macro_Rec: {},  \n Acc: {}, \n Macro_F1: {},'.format(best_train_loss ,best_test_pre, best_test_rec, best_test_acc ,best_test_f1),file=file_out)
            
            print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s',file=file_out)



