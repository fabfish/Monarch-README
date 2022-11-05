import torch

a = torch.Tensor((([1,2,3],[4,5,6]),([1,2,3],[4,5,6]))).transpose(-1,-2)

print(a)