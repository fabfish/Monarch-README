import torch
import torch.nn.functional as F

w = torch.Tensor(((1,2,3),(4,5,6),(7,8,9)))
print(w)
print(w.shape)

x = torch.Tensor((1,2,3))
linear = F.linear(x, w)

print(linear)