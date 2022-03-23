# 作者：York
# 时间：2022/3/22 19:50
import os

from scipy.io import loadmat

import preprocess
import loaddata
import numpy as np
import torch
import itertools
seed = 1
torch.manual_seed(seed)
import argparse
parser = argparse.ArgumentParser()
def load():
    x = loadmat(os.path.join('../data/final/douban.mat'))
    return (x['online_edge_label'][0][1],x['online_node_label'],x['offline_edge_label'][0][1],x['offline_node_label'],x['ground_truth'].T,x['H'])

parser.add_argument('--setup', type=int, default=2)
parser.add_argument('--dataset', type=str, default='douban')
parser.add_argument('--use_config', type=bool, default=True)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--transformer', type=int, default=1)
parser.add_argument('--prior_rate', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_wd', type=float, default=0.01)
parser.add_argument('--lr_recon', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=1)
args = parser.parse_args()

if args.use_config:
	try:
		import json
		f = open('../configs/%s.config' % args.dataset, 'r')
		arg_dict = json.load(f)
		for t in arg_dict:
			args.__dict__[t] = arg_dict[t]
	except:
		print('Error in loading config and use default setting instead')
print(type(args))
print(args)

a1, f1, a2, f2, ground_truth, prior = load()

feature_size = f1.shape[1]
#print(type(feature_size))
ns = [a1.shape[0], a2.shape[0]]

#print(type(ns), ns)
edge_1 = torch.LongTensor(np.array(a1.nonzero()))
print("a1的类型为：{}".format(type(a1)))
print("edge_1为：{}; edge_1的大小为:{}".format(edge_1,edge_1.shape))
edge_2 = torch.LongTensor(np.array(a2.nonzero()))
print("edge_2为：{}; edge_2的大小为:{}".format(edge_2,edge_2.shape))

ground_truth = torch.tensor(np.array(ground_truth, dtype=int)) - 1  # Original index start from 1
print(ground_truth,ground_truth.shape)
features = [torch.FloatTensor(f1.todense()), torch.FloatTensor(f2.todense())]
print("features 为{}，features的类型为".format(features))
edges = [edge_1, edge_2]
prior = torch.FloatTensor(prior)
prior_rate = args.prior_rate

#print("a1 是 {},a1的类型为{}".format(a1,type(a1)))
#print("f1 是 {},f1的类型为{}".format(f1,type(f1)))
#print("a2 是 {},a2的类型为{}".format(a2,type(a2)))
#print("f2 是 {},f2的类型为{}".format(f2,type(f2)))
#print("prior 是 {},prior的类型为{}".format(prior,type(prior)))
#print("prior 是 {},prior的类型为{}".format(prior,type(prior)))