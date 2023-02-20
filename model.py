import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #卷积层
        self.conv1=nn.Conv2d(1,32,kernel_size=(3,3),stride=(1,1))
        self.conv2=nn.Conv2d(32,64,kernel_size=(3,3),stride=(1,1))
        #Dropout层
        self.dropout1=nn.Dropout(0.2)
        #全连接
        self.fc1=nn.Linear(64*12*12,128)
        self.fc2=nn.Linear(128,10)
    
    def forward(self,x):
        #[bs,1,28,28]->[bs,32,26,26]
        x1=self.conv1(x)
        x1=F.relu(x1)
        #[bs,1,26,26]->[bs,64,24,24]
        x2=self.conv2(x1)
        x2=F.relu(x2)
        # [b, 64, 24, 24] -> [b, 64, 12, 12]
        x3=F.max_pool2d(x2, 2)
        x3=self.dropout1(x3)
        # [b, 64, 12, 12] => [b, 64 * 12 * 12] => [b, 9216]
        x4 = torch.flatten(x3, 1)
        fc=self.fc1(x4)
        fc=F.relu(fc)
        fc_2=self.fc2(fc)
        out=F.log_softmax(fc_2,dim=1)
        return out