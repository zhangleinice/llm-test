import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 检测是否可以使用 CUDA 加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# 加载 FashionMNIST 数据集，分为训练集和测试集
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

# 设置学习率、批大小和训练轮数
learning_rate = 1e-3
batch_size = 64
epochs = 3

# 创建数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 定义神经网络模型


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 将模型移至指定设备（GPU 或 CPU）
model = NeuralNetwork().to(device)

# 定义训练循环


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 获取数据集大小
    model.train()  # 将模型设置为训练模式
    for batch, (X, y) in enumerate(dataloader, start=1):  # 遍历数据加载器
        X, y = X.to(device), y.to(device)  # 将数据移至指定设备（GPU 或 CPU）
        # 计算预测和损失
        pred = model(X)  # 获取模型的预测
        loss = loss_fn(pred, y)  # 计算损失
        # 反向传播和权重更新
        optimizer.zero_grad()  # 梯度清零，以避免累积梯度
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 更新权重

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 定义测试循环


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 获取数据集大小
    num_batches = len(dataloader)  # 获取批次数量
    test_loss, correct = 0, 0  # 初始化测试损失和正确预测的数量

    model.eval()  # 将模型设置为评估模式，不进行梯度计算
    with torch.no_grad():  # 禁用梯度计算上下文管理器
        for X, y in dataloader:  # 遍历数据加载器
            X, y = X.to(device), y.to(device)  # 将数据移至指定设备
            pred = model(X)  # 获取模型的预测
            test_loss += loss_fn(pred, y).item()  # 计算测试损失
            correct += (pred.argmax(dim=-1) ==
                        y).type(torch.float).sum().item()  # 计算正确预测的数量

    test_loss /= num_batches  # 计算平均测试损失
    correct /= size  # 计算正确预测的百分比
    # 打印测试结果
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 使用交叉熵损失函数和 AdamW 优化器进行训练
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 主训练循环
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")

# 通过 3 轮迭代 (Epoch)，模型在训练集上的损失逐步下降、在测试集上的准确率逐步上升，证明优化器成功地对模型参数进行了调整，而且没有出现过拟合。
