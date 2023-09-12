import torch

x = torch.tensor([2.])
y = torch.tensor([3.])
z = (x + y) * (y - 2)

print(z)
# tensor([5.])


m = torch.tensor([1, 2, 3], dtype=torch.double)
n = torch.tensor([4, 5, 6], dtype=torch.double)

print(m + n)
# tensor([5., 7., 9.], dtype=torch.float64)
print(m - n)
# tensor([-3., -3., -3.], dtype=torch.float64)


# 形状转换 view：进行 view 操作的张量必须是连续的 (contiguous)
k = torch.tensor([1, 2, 3, 4, 5, 6])
print(k.view(2, 3))
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(k.view(3, 2))
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])


# reshape 函数，它与 view 功能几乎一致，并且能够自动处理非连续张量。
j = torch.tensor([1, 3, 5, 7, 9, 11])
print(j.reshape(3, 2))


# 张量形状不同，Pytorch 会自动执行广播
f = torch.arange(1, 4).view(3, 1)
print(f)
# tensor([[1],
#         [2],
#         [3]])
g = torch.arange(4, 6).view(1, 2)
print(g)
# tensor([[4, 5]])

print(f + g)
# tensor([[5, 6],
#         [6, 7],
#         [7, 8]])


# 索引与切片
x = torch.arange(12).view(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
print(x[1, 3])
# tensor(7)
print(x[1])
# tensor([4, 5, 6, 7])
print(x[1:3])
# tensor([[ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

print(x[:, 2])  # all elements in column 2)
# tensor([ 2,  6, 10])

print(x[:, 2:4])  # elements in column 2 & 3
# tensor([[ 2,  3],
#         [ 6,  7],
#         [10, 11]])

x[:, 2:4] = 100  # set elements in column 2 & 3 to 100
print(x)
# tensor([[  0,   1, 100, 100],
#         [  4,   5, 100, 100],
#         [  8,   9, 100, 100]])

# 降维与升维
a = torch.tensor([1, 2, 3, 4])
b = torch.unsqueeze(a, dim=0)
c = b.squeeze()

print(b, b.shape)
print(c, c.shape)
