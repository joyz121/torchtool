import os
import torchvision
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from model import *
from test import test
import onnx
import onnxruntime
import numpy as np
FILE = Path(__file__).resolve() #current directory
ROOT = FILE.parents[0]  #root directory
model_save_path=os.path.join(ROOT,'model.pt')
def get_data(batch_size):
    """获取数据"""

    # 获取测试集
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                       ]))
    train_loader = DataLoader(train, batch_size=batch_size)  # 分割测试集

    # 获取测试集
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch_size)  # 分割训练

    # 返回分割好的训练集和测试集
    return train_loader, test_loader


def train(model, epoch, train_loader,learning_rate):
    """训练"""

    # 训练模式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器

    # 迭代
    for step, (x, y) in enumerate(train_loader):
        # 加速
        if torch.cuda.is_available():
            model = model.cuda()
            x, y = x.cuda(), y.cuda()

        # 梯度清零
        optimizer.zero_grad()

        output = model(x)

        # 计算损失
        loss = F.nll_loss(output, y)

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 打印损失
        if step % 50 == 0:
            print('Epoch: {}, Step {}, Loss: {}'.format(epoch, step, loss))
def main():
    # 获取数据
    train_loader, test_loader = get_data(batch_size=64)
    iteration_num = 1  # 迭代次数
    network = Net()  # 实例化网络
    # 迭代
    for epoch in range(iteration_num):
        print("\n================ epoch: {} ================".format(epoch))
        train(network, epoch, train_loader,learning_rate=0.001)
        test(network, test_loader)
    # save model
    torch.save(network,model_save_path)

if __name__ == "__main__":
    main()
    #调用onnx 计算正确率
    # correct = 0
    # seesion=onnxruntime.InferenceSession('./model.onnx')
    # input_name=seesion.get_inputs()[0].name
    # output_name =seesion.get_outputs()[0].name
    # train_loader,test_loader = get_data(batch_size=1)
    # for x, y in test_loader:
    #     x=np.array(x)
    #     output=seesion.run([output_name],{input_name:x})
    #     output=np.array(output)
    #     pred = output.argmax()
    #         # 计算准确个数
    #     if pred==y:
    #         correct += 1
    #     # 计算准确率
    # accuracy = correct / len(test_loader.dataset) * 100

    # # 输出准确 
    # print("Test Accuracy: {}%".format(accuracy))