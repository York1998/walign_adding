import numpy as np
import os
from scipy.io import loadmat
import torch
import torch_geometric


def load_final(dataset_name):
	x = loadmat(os.path.join('data', 'final', '{name}.mat'.format(name=dataset_name)))
	if dataset_name=='douban':
		return (x['online_edge_label'][0][1], 
			x['online_node_label'], 
			x['offline_edge_label'][0][1], 
			x['offline_node_label'],
			x['ground_truth'].T,
			x['H'])

def load_arena(dataset_name, noise=0):
	n = 1135
	edges = torch.tensor(np.loadtxt('data/arena/arenas_combined_edges.txt'), dtype=torch.long).transpose(0,1).squeeze()
	row,col = edges
	edge2 = torch_geometric.utils.to_undirected(edges[:, (edges[0, :] >= n).nonzero().view(-1) ] - n).clone().squeeze()
	edge1 = torch_geometric.utils.to_undirected(edges[:, (edges[0, :] < n).nonzero().view(-1) ]).clone().squeeze()
	try:
		feature = np.load('data/arena/attr1-2vals-prob%f.npy' % noise)
		feature = torch.tensor(feature)
	except:
		feature = np.load('data/arena/attr1-2vals-prob0.000000', allow_pickle=True)
		feature = torch.tensor(feature)
		ft_sel = torch.randperm(feature.size(0))[:int(noise * feature.size(0))]
		feature[ft_sel] = 1 - feature[ft_sel]
		np.save('data/arena/attr1-2vals-prob%f' % noise, feature.numpy())
	ft_rich = torch.rand(2, 50, dtype=torch.float)
	ft_rich = ft_rich[feature.view(-1)]

	feature1 = ft_rich[:n, :]
	feature2 = ft_rich[n:, :]
	ledge = edge2.size(1)
	perm_mapping = torch.tensor(np.loadtxt('data/arena/arenas_mapping.txt'), dtype=torch.long).transpose(0, 1)
	return edge1, feature1, edge2, feature2, perm_mapping
	
def load_dbp(dataset_name, language='en_fr'):
	dataset = torch_geometric.datasets.DBP15K('../data/dbp15k', language)
	edge1 = dataset[0].edge_index1
	edge2 = dataset[0].edge_index2
	feature1 = dataset[0].x1.view(dataset[0].x1.size(0), -1)
	feature2 = dataset[0].x2.view(dataset[0].x2.size(0), -1)
	ground_truth = torch.cat( (dataset[0].train_y, dataset[0].test_y), dim=-1)
	return edge1, feature1, edge2, feature2, ground_truth


"""
	装载数据集，返回
"""
def load_geometric(dataset_name, noise_level = 0, noise_type = 'uniform'):
	if dataset_name == 'ppi':
		"下载的地址：url = 'https://data.dgl.ai/dataset/ppi.zip'"
		""" 描述如下：
		
		‘The protein-protein interaction networks 
		from the “Predicting Multicellular Function through Multi-layer Tissue Networks” paper, 
		containing positional gene sets, 
		motif gene sets and immunological signatures as features (50 in total) 
		and gene ontology sets as labels (121 in total).’ 
		
		还有一些数据集：https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
		"""

		"""
		" 第一个参数是 root：存储位置：Root directory where the dataset should be saved." \
		" 第二个参数是 split：split (string) – If train, loads the training dataset. If val, loads the validation dataset. If test, loads the test dataset. (default: train) "
		" 第三个参数是： transform： A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version. The data object will be transformed before every access. (default: None)"
		" 第四个参数是：pre_transform：A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version. The data object will be transformed before being saved to disk. (default: None)"
		" 第五个参数是：pre_filter (callable, optional) – A function that takes in an torch_geometric.data.Data object and returns a boolean value, indicating whether the data object should be included in the final dataset. (default: None)"
		"""
		"以下只有一个参数，即 root"
		dataset = torch_geometric.datasets.PPI(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'ppi'))

	"""
		返回的形式如下：
			{
				data : x=[44906,50] , edge_index = [2,1226368] , y=[44906,121]
				function : map ; concat
				num_classes = 121
				num_edge_features = 0
				num_features = 50
				num_node_features = 50
				...
			}
			
	"""
	edge1 = dataset[0].edge_index # 获取ppi的边关系，一般有两种，一种是(id,id)的列表，一种是邻接表，在这里应该是邻接表
	feature1 = dataset[0].x # 获取ppi的节点属性x
	edge2 = edge1.clone() # 克隆出另外一个图，用于图对齐后面的hit@k指标计算
	ledge = edge2.size(1) # 图2
	" torch.randperm(n) 返回一个0到n-1的数组 ,此数组是随机的，即不一定按照0到n-1进行排序 "
	""" 
		x[:,n]表示在全部数组中取第n个数据，即取所有集合的第n个数据：
		例如： X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])  
		print X[:,0]  #输出：[0 2 4 6 8 10 12 14 16 18]
		
		x[n,:]表示在n个数组中去全部数据，即取第n个集合的的所有数据：
		例如： X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]])  
		print X[1,:]  #输出：[2 3]

		扩展：x[:,m:n]：即取所有数据集的第m到n-1列数据
		X = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14],[15,16,17],[18,19,20]])  
		print X[:,1:3] #输出：[[ 1 2]
								4 5]
								7 8]
								10 11]
								13 14]
								16 17]
								19 20]]
		
	"""
	edge2 = edge2[:, torch.randperm(ledge)[:int(ledge*0.9)]] #将数据打乱，并取90%的数据用来作为图2的节点
	perm = torch.randperm(feature1.size(0))
	perm_back = torch.tensor(list(range(feature1.size(0))))
	perm_mapping = torch.stack([perm_back, perm])
	edge2 = perm[edge2.view(-1)].view(2, -1) 
	edge2 = edge2[:, torch.argsort(edge2[0])]
	feature2 = torch.zeros(feature1.size())
	feature2[perm] = feature1.clone()
	if noise_type == 'uniform':
		feature2 = feature2 + 2 * (torch.rand(feature2.size())-0.5) * noise_level
	elif noise_type == 'normal':
		feature2 = feature2 + torch.randn(feature2.size()) * noise_level
	return edge1, feature1, edge2, feature2, perm_mapping


"""
	装载数据集
"""
def load(dataset_name='cora', noise_level=0):
	if dataset_name in ['ppi']:
		" ppi数据集存在于torch_geometric，因此可以用此函数直接进行导入； "
		return load_geometric(dataset_name, noise_level=noise_level, noise_type='uniform')
	elif dataset_name in ['douban']:
		return load_final(dataset_name)
	elif dataset_name in ['arena']:
		return load_arena(dataset_name, noise=noise_level)