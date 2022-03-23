import numpy as np
import scipy.sparse as sp
import torch
'''
    utils：定义了加载数据等工具性的函数
    layers：定义了模块如何计算卷积
    models：定义了模型train
    train：包含了模型训练信息
'''

'''
在很多的多分类问题中，特征的标签通常都是不连续的内容（如本文中特征是离散的字符串类型）
，为了便于后续的计算、处理，需要将所有的标签进行提取，并将标签映射到一个独热码向量中。
'''
def encode_onehot(labels):
    # 将所有的标签整合成一个重复的列表
    classes = set(labels) # set函数创建一个无序不重复元素集
    '''
        enumerate() 函数生成序列，带有索引 i 和 值 c 。
        这一句将string 类型的lable 变为int类型的lable , 建立映射关系
        np.identity（len(classes)) 为创建一个 Classes的单位矩阵
        创建一个字典 , 索引为 lable , 值为独热码向量 (就是之前生成矩阵中的某一行)
    '''
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    '''
        x:xxx for a,b in yyy解释如下:
        yyy是一个容器,它里面有成对的a,b
        会分别赋值给x:xxx 
        这里的意思就是生成了单位矩阵之后 i 是每个Class中lable的下标，c是lable对应的val
        也就是说c值和他对应的那一行(他所对应的那一行)词向量矩阵进行词典
    '''
    #print(classes_dict)
    # [i, :] 就是说保留第i行的所有列, 也就是说 将生成的单位矩阵的的第i行和c进行一个字典
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # map(function, iterable)是对指定序列iterable中的每一个元素调用function函数
    # 每个labels 会对应调用其字典的值来与之匹配
    # 这句话也就是说将输入一一对应 one - hot 编码进行输出
    '''
        example:
        #  map(lambda x: x ** 2, [1, 2, 3, 4, 5])
        #  output:[1, 4, 9, 16, 25]
    '''
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """  加载引文网络数据集（目前仅限cora） """
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵 和 COO矩阵一样结构
    # [:, 1:-1]是指行全部选中、列选取第二列至倒数第二列，float32类型
    # 这句功能就是去除论文样本的编号和类别，留下每篇论文的词向量，并将稀疏矩阵编码压缩
    # 提取论文样本的类别标签，并将其转换为one-hot编码形式
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 提取论文样本的编号id数组
    idx_map = {j: i for i, j in enumerate(idx)}
    # enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), # flatten：降维，返回一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    '''
        因为之前建图是无向图，这里要建立有向图。
        第一个参数是data矩阵，这里之所以全是1是因为我们是连通性的邻接矩阵，如果有边的话那么就是1
        (i,j) 就是边的两个节点信息
        因为是对标签之间的图，所以大小就是一个标签数量的方阵
    '''
    # 建立对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj_2 = adj + adj.T.multiply(adj.T > adj) 和上述式子等价为了产生对称的矩阵
    # adj_3 = adj + adj.T
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # 对称邻接矩阵+单位矩阵，并进行归一化
    # 这里即是A+I，添加了自连接的邻接矩阵
    # adj=D^-1(A+I)

    # 分割为train，val，test三个集，最终数据加载为torch的格式并且分成三个数据集
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense())) # 将特征矩阵转化为张量形式
    labels = torch.LongTensor(np.where(labels)[1])
    # np.where(condition)，输出满足条件condition(非0)的元素的坐标，np.where()[1]则表示返回列的索引、下标值
    # 说白了就是将每个标签one-hot向量中非0元素位置输出成标签
    # one-hot向量label转常规label：0,1,2,3,……
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 将scipy稀疏矩阵转换为torch稀疏张量，具体函数下面有定义
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    # 转化为张量
    return adj, features, labels, idx_train, idx_val, idx_test
    # 返回（样本关系的对称邻接矩阵的稀疏张量，样本特征张量，样本标签，
    #		训练集索引列表，验证集索引列表，测试集索引列表）

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # .sum(1)意思是对第一维度进行求和 也就是对每一行求和
    r_inv = np.power(rowsum, -1).flatten() # 对rowsum 求和之后的倒数 在行方向上进行降维。等价于 .flatten('A')
    r_inv[np.isinf(r_inv)] = 0. # 0倒数是inf所以要把inf的地方全部变为0
    r_mat_inv = sp.diags(r_inv)  # 创建对角矩阵 将已经变成一维的 矩阵再次变成一个方形的矩阵
    '''
        example: [3,2] 的一个ones矩阵
        [[1 1]
        [1 1]
        [1 1]]  np.power(rowsum, 2).flatten()之后变成了一个
        [4 4 4]
        sp.diags 变成了：
        (0, 0)	4.0
        (1, 1)	4.0
        (2, 2)	4.0
        左边是(x,y)的tuple坐标 右边则是他对应的val
    '''
    mx = r_mat_inv.dot(mx) # 进行点乘 也就是论文中对应的公式D^-1A而不是计算论文中的D^-1/2AD^-1/2
    return mx


def accuracy(output, labels): #  准确率，此函数可参考学习借鉴复用
    '''
    我们在计算准确率的时候经常会看到
    pred_y = torch.max(predict, 1)[1].numpy()
    y_label = torch.max(label, 1)[1].data.numpy()
    accuracy = (pred_y == y_label).sum() / len(y_label)
    这样的结构 下面的结构和这里的结构类似
    '''
    preds = output.max(1)[1].type_as(labels)
    # (1) 代表的是第一个维度的最大值 ，[1]代表的是这个最大值对应的下标
    # type_as(x) 转换为相同类型(x) ex: 如果 preds 是 int 类型但是 labels是float类型，那么就会吧preds转换为float类型
    correct = preds.eq(labels).double()
    '''
        example:
        outputs=torch.FloatTensor([[1],[2],[3]])
        targets=torch.FloatTensor([[0],[2],[3]])
        targets.eq(outputs.data).double()之后的结果是：
        tensor([[0.],
        [1.],
        [1.]], dtype=torch.float64)
        targets.eq(outputs.data).double().sum() 之后的结果是：
        tensor(2., dtype=torch.float64)
    '''
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量。"""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # tocoo()是将此矩阵转换为Coo格式，astype()转换数组的数据类型
    '''
        COO 矩阵的 例子：
        [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]]
        对应的COO 矩阵入下：
        (0,0) 1.0
        (1,1) 1.0
        (2,2) 1.0
    '''
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # vstack()将两个数组按垂直方向堆叠成一个新数组
    # torch.from_numpy()是numpy中的ndarray转化成pytorch中的tensor
    # Coo的索引
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

