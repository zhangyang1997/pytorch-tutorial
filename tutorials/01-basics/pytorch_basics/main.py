import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                               目录                                 #
# ================================================================== #

# 1. autograd计算梯度举例1                    (Line 25 to 39)
# 2. autograd计算梯度举例2                    (Line 46 to 83)
# 3. 从numpy加载数据                          (Line 90 to 97)
# 4. 输入pipline                             (Line 104 to 129)
# 5. 自定义数据的输入pipline                  (Line 136 to 156)
# 6. 预定义模型                              (Line 163 to 176)
# 7. 保存和加载模型                          (Line 183 to 189)


# ================================================================== #
#                     1. autograd计算梯度举例1                        #
# ================================================================== #

# 创建张量
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# 创建计算图
y = w * x + b    # y = 2 * x + 3

# 计算梯度
y.backward()

# 输出梯度
print(x.grad)
print(w.grad)
print(b.grad)

'''
x.grad = tensor(2.)
w.grad = tensor(1.)
b.grad = tensor(1.)
'''


# ================================================================== #
#                    2. autograd计算梯度举例2                         #
# ================================================================== #

# 创建10×3和10×2的两个随机张量
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 构建两个全连接层
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

'''
w:  Parameter containing:
tensor([[-0.0707,  0.2341,  0.4827],
        [-0.5092, -0.1537,  0.2582]], requires_grad=True)
b:  Parameter containing:
tensor([ 0.5335, -0.2167], requires_grad=True)
'''

# 构建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss: ', loss.item())

'''
loss:  1.831163763999939
'''

# 反向传播
loss.backward()

# 输出梯度
print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)
'''
dL/dw:  tensor([[ 0.5340,  0.4947,  0.1947],
        [-0.1455,  0.5270,  0.6877]])
dL/db:  tensor([ 0.5586, -0.8556])
'''

# 1步梯度下降
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 打印出1步梯度下降后的损失
pred = linear(x)
loss = criterion(pred, y)
print('1步优化后的损失: ', loss.item())
'''
1步优化后的损失:  1.631872534751892
'''


# ================================================================== #
#                     3. 从numpy加载数据                              #
# ================================================================== #

# 创建一个numpy数组
x = np.array([[1, 2], [3, 4]])

# 将numpy数组转换为张量
y = torch.from_numpy(x)

# 将张量转换为numpy数组
z = y.numpy()


# ================================================================== #
#                         4. 输入pipeline                           #
# ================================================================== #

# 下载并构建CIFAR-10数据集.
train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# 获取一对数据(从磁盘读数据)
image, label = train_dataset[0]
print(image.size())
print(label)
'''
torch.Size([3, 32, 32])
6
'''

# 数据加载器(提供队列和线程的方法).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

# 迭代开始，队列和线程开始加载数据
data_iter = iter(train_loader)

# 小批量图像和标签.
images, labels = data_iter.next()

# 数据加载器的实际使用情况
for images, labels in train_loader:
    # 训练代码写在此处
    pass


# ================================================================== #
#                5. 自定义数据集的输入pipeline                         #
# ================================================================== #

# 构建自定义数据集
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. 初始化文件路径或者文件名列表
        pass

    def __getitem__(self, index):
        # TODO
        # 1. 从文件中读取一个数据(例如numpy.fromfile, PIL.Image.open).
        # 2. 预处理数据(例如torchvision.Transform).
        # 3. 返回数据对(例如image and label).
        pass

    def __len__(self):
        # 返回数据集大小
        return 0


# 使用预构建的数据加载器
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)


# ================================================================== #
#                        6. 预训练模型                                #
# ================================================================== #

# 下载并加载预训练的ResNet-18模型.
resnet = torchvision.models.resnet18(pretrained=True)

# 如果只想微调模型的顶层，请进行如下设置.
for param in resnet.parameters():
    param.requires_grad = False

# 更换顶层以进行微调.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# 前向计算
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())
'''
64x3x224x224->64x100
torch.Size([64, 100])
'''

# ================================================================== #
#                      7. 保存并加载模型                              #
# ================================================================== #

# 保存并加载整个模型
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# 仅保存和加载模型参数（推荐）
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
