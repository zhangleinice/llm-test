import torch
import torchvision.models as models


# 保存和加载模型权重

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

# we do not specify pretrained=True, i.e. do not load default weights
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# 保存和加载完整模型

model = models.vgg16(pretrained=True)
torch.save(model, 'model.pth')

model = torch.load('model.pth')
