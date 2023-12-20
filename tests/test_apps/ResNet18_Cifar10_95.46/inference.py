import torch
import torch.nn as nn
from utils.readData import read_dataset
from utils.ResNet import ResNet18
import numpy as np
import time
import sys
import ctypes
import os

# load remoting bottom library
path = os.getenv('REMOTING_BOTTOM_LIBRARY')
cpp_lib = ctypes.CDLL(path)
start_trace = cpp_lib.startTrace

if(len(sys.argv) != 2):
    print('Usage: python3 inference.py num_iter')
    sys.exit()

num_iter = int(sys.argv[1])

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 1
train_loader,valid_loader,test_loader = read_dataset(batch_size=batch_size,pic_path='dataset')
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
model.eval()  # 验证模型

# remove initial overhead
for i in range(20):
    data, target = next(iter(test_loader))
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(output, 1)    
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.data.cpu().numpy())

start_trace()

T1 = time.time()

for i in range(num_iter):
    data, target = next(iter(test_loader))
    data = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data).to(device)
    # convert output probabilities to predicted class(将输出概率转换为预测类)
    _, pred = torch.max(output, 1)    
    # compare predictions to true label(将预测与真实标签进行比较)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.data.cpu().numpy())
    
T2 = time.time()
print('time used: ', T2-T1)
print(correct)
