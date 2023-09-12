import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# 定义神经网络模型


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # 将输入数据展平为一维张量
        self.flatten = nn.Flatten()

        # 创建一个包含多个线性层和激活函数的序列
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  # 输入大小 28x28，输出大小 512
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(512, 256),  # 输入大小 512，输出大小 256
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Linear(256, 10),   # 输入大小 256，输出大小 10（对应于类别数量）
            nn.Dropout(p=0.2)  # 使用丢弃层，防止过拟合
        )

    def forward(self, x):
        # 展平输入数据
        x = self.flatten(x)

        # 将数据传递通过线性层和激活函数的序列
        logits = self.linear_relu_stack(x)

        return logits


# 创建神经网络模型并将其移到适当的设备（CPU 或 CUDA）
model = NeuralNetwork().to(device)
# print(model)


# 创建一个形状为 (4, 28, 28) 的随机张量 X，并将其移动到指定的设备（CPU 或 CUDA）
X = torch.rand(4, 28, 28, device=device)

# 使用定义的神经网络模型对输入数据 X 进行前向传播，得到 logits
logits = model(X)

# 使用 Softmax 函数对 logits 进行处理，以获得类别概率分布
pred_probab = nn.Softmax(dim=1)(logits)

# 打印预测概率张量的大小
print(pred_probab.size())

# 从概率张量中选择具有最高概率的类别作为预测结果
y_pred = pred_probab.argmax(-1)

# 打印预测的类别
print(f"Predicted class: {y_pred}")
