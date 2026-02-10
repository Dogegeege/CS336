import torch

x = torch.tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
y = torch.tensor([[1, 2]])
z = torch.tensor([[[[4], [5], [6]], [[7], [8], [9]]]])
print(x[y] == z)
print(z.shape)

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
