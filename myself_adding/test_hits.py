# 作者：York
# 时间：2022/3/28 16:02
import torch
import torch.nn.functional as F

import graphmodel

net = graphmodel.LGCN(4, 4)
print(net)