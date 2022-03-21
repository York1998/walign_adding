# 作者：York
# 时间：2022/3/10 22:30
import os.path

import torch
import torch_geometric
import numpy as np
dataset = torch_geometric.datasets.PPI(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','data','ppi'))
edge1 = dataset[0].edge_index
print(type(edge1))
ledge = edge1.size(1)
edge2 = edge1.clone()


x = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
#print(x[:,4])

X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])
print(X[:,1])
print(X[1,:])

print(torch.randperm(ledge))
print(torch.randperm(ledge)[:int(ledge*0.9)])
edge2 = edge2[:, torch.randperm(ledge)[:int(ledge*0.9)]]
print(edge2)