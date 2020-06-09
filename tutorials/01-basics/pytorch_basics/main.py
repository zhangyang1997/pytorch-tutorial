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
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. autograd计算梯度举例2                         #
# ================================================================== #

# 创建10×3和10×2的两个随机张量
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# 构建两个全连接层
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# 构建损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# 前向传播
pred = linear(x)

# 计算损失
loss = criterion(pred, y)
print('loss: ', loss.item())

# 反向传播
loss.backward()

# 输出梯度
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1步梯度下降
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# 打印出1步梯度下降后的损失
pred = linear(x)
loss = criterion(pred, y)
print('1步优化后的损失: ', loss.item())


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
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# 获取一对数据(从磁盘读数据.
image, label = train_dataset[0]
print (image.size())
print (label)

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
        # 1. 从文件中读取一个数据(e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
