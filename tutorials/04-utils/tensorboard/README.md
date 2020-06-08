# TensorBoard in PyTorch

在本教程中，我们使用简单的神经网络实现一个MNIST分类器，并使用[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)可视化训练过程。在训练阶段，我们通过scalar_summary绘制loss and accuracy函数，通过image_summary将训练图像可视化。此外，我们使用‘histogram_summary’来可视化神经网络参数的权值和梯度值。可以在[此处](https://github.com/yunjey/pytor-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py #L81-L97)。找到处理这些摘要函数的PyTorch代码

![alt text](gif/tensorboard.gif)



## 用法

#### 1. 安装依赖
```bash
$ pip install -r requirements.txt
```

#### 2. 训练网络
```bash
$ python main.py
```

#### 3. 打开TensorBoard
要运行TensorBoard，打开一个新的终端并运行下面的命令。然后，在web浏览器上打开http://localhost:6006/。

```bash
$ tensorboard --logdir='./logs' --port=6006
```
