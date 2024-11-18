import torch
import torch.nn as nn

# 创建numpy数组时没有指定dtype,默认使用int64,从numpy array转成torch.Tensor后, 数据类型变成了Long
inputs = torch.tensor([1, 2, 3], dtype = torch.float32)
targets = torch.tensor([1, 2, 5], dtype = torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss(reduction='sum')
result = loss(inputs, targets)

# 误差平方损失函数
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)

# 交叉熵损失函数
x = torch.tensor([0.1, 0.2, 0.3, 0.4])
y = torch.tensor([3])
x = torch.reshape(x, (1, 4))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result)
