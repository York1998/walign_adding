import math

import torch

from torch.nn.parameter import Parameter  # 可以用parameter()函数
from torch.nn.modules.module import Module  # 定义网络层的模块

'''
parameter()将一个不可训练的类型Tensor转换成可以
训练的类型parameter并将其绑定到这个module里面，
所以经过类型转换这个就变成了模型的一部分，成为了
模型中根据训练可以改动的参数了。使用这个函数的目
的也是想让某些变量在学习的过程中不断的修改其值以
达到最优化。
'''


class GraphConvolution(Module):
    '''
    简单的GCN层，类似于https://arxiv.org/abs/1609.02907

	参数：
		in_features：输入特征，每个输入样本的大小
		out_features：输出特征，每个输出样本的大小
		bias：偏置，如果设置为False，则层将不会学习加法偏差。默认值：True
	属性：
		weight：形状模块的可学习权重（out_features x in_features）
		bias：形状模块的可学习偏差（out_features）
	'''

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        '''
        # 常见用法self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))：
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter
        # 绑定到这个module里面，所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化

        # 先转化为张量，再转化为可训练的Parameter对象
        # Parameter用于将参数自动加入到参数列表
        '''
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            # 第一个参数必须按照字符串形式输入
        # 将Parameter对象通过register_parameter()进行注册
        # 为模型添加参数
        self.reset_parameters()

    def reset_parameters(self):  # 参数随机初始化函数
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数  size（1）为行
        # size包括(in_features, out_features)，size(1)应该是指out_features
        # stdv=1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)  # uniform() 方法将随机生成下一个实数，它在 [-stdv, stdv] 范围内
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布随机初始化

    '''
       前馈运算 即计算A~ X W(0)
       input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
       直接输入与权重之间进行torch.mm操作，得到support，即XW
       support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    '''

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # input和self.weight矩阵相乘
        # torch.mm(a, b)是矩阵a和b矩阵相乘，torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        output = torch.spmm(adj, support)
        # spmm()是稀疏矩阵乘法，说白了还是乘法而已，只是减小了运算复杂度
        # 最新spmm函数移到了torch.sparse模块下，但是不能用
        if self.bias is not None:
            return output + self.bias  # 返回（系数*输入*权重+偏置）
        else:
            return output  # 返回（系数*输入*权重）无偏置

    # 通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，#0.01，0.94]，里面的值代表该x对应标签不同的概率，故此值可转换为#[0,0,0,0,0,0,1]，对应我们之前把标签onthot后的第七种标签

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
        # 打印形式是：GraphConvolution (输入特征 -> 输出特征)
