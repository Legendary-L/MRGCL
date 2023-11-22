
import networkx as nx

from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader

import dgl

import numpy as np
from sklearn.neighbors import kneighbors_graph

import torch.nn as nn

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import degree

# 在这之后定义HANModel类和其他相关代码


from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy


import numpy as np

import os

os.environ['DGL_DOWNLOAD_DIR'] = '/data/lab106/wby/MRCGNN/'


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)








class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features, adjacency_matrix):
        # GCN公式：A~ D^-0.5 A D^-0.5 X W
        # A 表示邻接矩阵，X 表示特征矩阵，W 表示权重矩阵
        # D 为度矩阵，定义为度数的对角矩阵
        # A~ 表示 A+I，其中 I 为单位矩阵
        # D^-0.5 表示 D 的 -0.5 次方

        # 计算 A~，即邻接矩阵加上自连接
        identity = torch.eye(adjacency_matrix.size(0))
        adjacency_matrix_hat = adjacency_matrix + identity

        # 计算度矩阵 D
        degree = torch.sum(adjacency_matrix_hat, dim=1)
        degree_sqrt_inv = torch.pow(degree, -0.5)
        degree_sqrt_inv[degree_sqrt_inv == float('inf')] = 0
        degree_sqrt_inv_matrix = torch.diag(degree_sqrt_inv)

        # 计算对称归一化后的邻接矩阵 A~
        symmetric_adjacency = torch.matmul(torch.matmul(degree_sqrt_inv_matrix, adjacency_matrix_hat), degree_sqrt_inv_matrix)

        # GCN 层计算
        transformed_features = torch.matmul(symmetric_adjacency, features)
        transformed_features = self.linear(transformed_features)
        transformed_features = F.relu(transformed_features)

        return transformed_features



def transform_features1(features, adjacency_matrix):
    # 基于节点级关注度的特征变换
    gcn_layer = GCNLayer(input_dim=features.shape[1], output_dim=features.shape[1])
    new_features = gcn_layer(features, adjacency_matrix)
    return new_features

def transform_features2(features, adjacency_matrix):
    # 基于节点级和层次级关注度的特征变换
    gcn_layer1 = GCNLayer(input_dim=features.shape[1], output_dim=features.shape[1]//2)  # 使用一半维度进行第一次变换
    intermediate_features = gcn_layer1(features, adjacency_matrix)
    gcn_layer2 = GCNLayer(input_dim=intermediate_features.shape[1], output_dim=features.shape[1])  # 恢复到原始维度
    new_features = gcn_layer2(intermediate_features, adjacency_matrix)
    return new_features





class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:,2]
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):


        return  (self.entity1[index], self.entity2[index], self.relationtype[index])


def load_data(args, val_ratio=0.1, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data

    drug_list = []
    with open('data/drug_listxiao.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            drug_list.append(row[0])

    #print(len(drug_list))

    zhongzi=args.zhongzi

    def loadtrainvaltest():
        #train dataset
        train=pd.read_csv('data/'+str(zhongzi)+'/ddi_training1xiao.csv')
        train_pos=[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(train_pos)
        train_pos = np.array(train_pos)
        for i in range(train_pos.shape[0]):
            train_pos[i][0] = int(drug_list.index(train_pos[i][0]))
            train_pos[i][1] = int(drug_list.index(train_pos[i][1]))
            train_pos[i][2] = int(train_pos[i][2])
        label_list=[]
        for i in range(train_pos.shape[0]):
            label=np.zeros((65))
            label[int(train_pos[i][2])]=1
            label_list.append(label)
        label_list=np.array(label_list)
        train_data= np.concatenate([train_pos, label_list],axis=1)

        #val dataset
        val = pd.read_csv('data/'+str(zhongzi)+'/ddi_validation1xiao.csv')
        val_pos = [(h, t, r) for h, t, r in zip(val['d1'], val['d2'], val['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(val_pos)
        val_pos= np.array(val_pos)
        for i in range(len(val_pos)):
            val_pos[i][0] = int(drug_list.index(val_pos[i][0]))
            val_pos[i][1] = int(drug_list.index(val_pos[i][1]))
            val_pos[i][2] = int(val_pos[i][2])
        label_list = []
        for i in range(val_pos.shape[0]):
            label = np.zeros((65))
            label[int(val_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        val_data = np.concatenate([val_pos, label_list], axis=1)

        #test dataset
        test = pd.read_csv('data/'+str(zhongzi)+'/ddi_test1xiao.csv')
        test_pos = [(h, t, r) for h, t, r in zip(test['d1'],test['d2'], test['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(test_pos)
        test_pos= np.array(test_pos)
        #print(test_pos[0])
        for i in range(len(test_pos)):
            test_pos[i][0] = int(drug_list.index(test_pos[i][0]))
            test_pos[i][1] = int(drug_list.index(test_pos[i][1]))
            test_pos[i][2] = int(test_pos[i][2])
        label_list = []
        for i in range(len(test_pos)):
            label = np.zeros((65))
            label[int(test_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        test_data = np.concatenate([test_pos, label_list], axis=1)
        #print(train_data.shape)
        #print(val_data.shape)
        #print(test_data.shape)
        return train_data,val_data,test_data

    train_data,val_data,test_data=loadtrainvaltest()
    params = {'batch_size': args.batch, 'shuffle': False, 'num_workers': args.workers, 'drop_last': False}


    training_set = Data_class(train_data)

    train_loader = DataLoader(training_set, **params)


    validation_set = Data_class(val_data)

    val_loader = DataLoader(validation_set, **params)


    test_set = Data_class(test_data)

    test_loader = DataLoader(test_set, **params)

    print('Extracting features...')



    features = np.load('trimnet/drug_emb_trimnet'+str(zhongzi)+'.npy')
    ids = np.load('trimnet/drug_idsxiao.npy')
    ids=ids.tolist()
    features1=[]
    for i in range(len(drug_list)):
        features1.append(features[ids.index(drug_list[i])])

    
    

    features=np.array(features1)
    print(type(features))
    
    
    
    # 使用K最近邻方法构建邻接矩阵
    k = 5  # 设置K值
    adjacency_matrix = kneighbors_graph(features, k, mode='connectivity', include_self=False)

    # 将邻接矩阵转换成稀疏矩阵的形式
    adjacency_matrix = adjacency_matrix.toarray()
    
    # 调用函数进行特征变换
    # 假设 features 是原始特征张量，adjacency_matrix 是邻接矩阵张量
    # 注意：features 和 adjacency_matrix 都需要转换成 PyTorch 的 Tensor 类型
    features_tensor = torch.from_numpy(features).float()
    adjacency_matrix_tensor = torch.from_numpy(adjacency_matrix).float()

    new_features1 = transform_features1(features_tensor, adjacency_matrix_tensor)
    new_features2 = transform_features2(features_tensor, adjacency_matrix_tensor)
    
    
    
    
    
    #new_features_o = normalize(new_features2)
    #args.dimensions = new_features_o.shape[1]
    new_x_o = torch.tensor(new_features2, dtype=torch.float)
    
    
    #new_features_a = normalize(new_features1)
    #args.dimensions = new_features_a.shape[1]
    new_x_a = torch.tensor(new_features1, dtype=torch.float)
    
    
    
    
    
    
    
    
    features_o = normalize(features)

    args.dimensions = features_o.shape[1]

    # adversarial nodes

    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]
    y_a = torch.cat((torch.ones(572, 1), torch.zeros(572, 1)), dim=1)
    x_o = torch.tensor(features_o, dtype=torch.float)
    positive1=copy.deepcopy(train_data)

    edge_index_o = []
    label_list = []
    label_list11 = []
    for i in range(positive1.shape[0]):

    #for h, t, r ,label in positive1:
        a = []
        a.append(int(positive1[i][0]))
        a.append(int(positive1[i][1]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        a = []
        a.append(int(positive1[i][1]))
        a.append(int(positive1[i][0]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        b = []
        b.append(int(positive1[i][2]))
        b.append(int(positive1[i][2]))
        label_list11.append(b)


    

    
    edge_index_o = torch.tensor(edge_index_o, dtype=torch.long)

    #data_o = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)
    data_o = Data(x=new_x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)
    #data_a = Data(x=new_x_a, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)
    data_s = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)
    
    
    
    x_a=torch.tensor(new_features1, dtype=torch.float)
    
    #x_a = torch.tensor(features_a, dtype=torch.float)
    #data_s = Data(x=x_a, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)




    random.shuffle(label_list11)
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    label_list11 = flatten(label_list11)
    #data_a = Data(x=x_o, y=y_a, edge_type=label_list11)
    data_a = Data(x=x_a, y=y_a, edge_type=label_list11)
    
    print('Loading finished!')
    return data_o, data_s, data_a, train_loader, val_loader, test_loader


