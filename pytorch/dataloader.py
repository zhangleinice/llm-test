from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import SequentialSampler, RandomSampler


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# 如果设置了 shuffle 参数，DataLoader 就会自动创建一个顺序或乱序的 sampler
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# img = train_features[0].squeeze()
# label = train_labels[0]

# print(img.shape)
# print(f"Label: {label}")


# Feature batch shape: torch.Size([64, 1, 28, 28])
# Labels batch shape: torch.Size([64])
# torch.Size([28, 28])
# Label: 8


train_sampler = RandomSampler(training_data)
test_sampler = SequentialSampler(test_data)

# 自定义sampler控制顺序
train_dataloader = DataLoader(
    training_data, batch_size=64, sampler=train_sampler)
test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
test_features, test_labels = next(iter(test_dataloader))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

# Feature batch shape: torch.Size([64, 1, 28, 28])
# Labels batch shape: torch.Size([64])
# Feature batch shape: torch.Size([64, 1, 28, 28])
# Labels batch shape: torch.Size([64])
