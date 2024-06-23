import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
convert=transforms.Compose([transforms.ToTensor()])
train_dataset=datasets.MNIST('',train=True,download=True,transform=convert)
test_dataset=datasets.MNIST('',train=False,download=True,transform=convert)

# for data in train_dataset:
#     print (data)
#     break

train_loader=DataLoader(train_dataset,batch_size=10,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=10,shuffle=False)

for data in train_loader:
    # print (data)
    # print(data[0].shape)
    break
# x,y=data[0],data[1]
# print(x.shape,y.shape)
# import matplotlib.pyplot as plt  # pip install matplotlib

# plt.imshow(data[0][0].view(28,28))
# plt.show()


# counter_dict={0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
# total=0
# for data in train_loader:
#     x,y=data[0],data[1]
#     for z in y:
       
#         total+=1
#         counter_dict[int(z)]+=1
# print(total)
# print(counter_dict)
# for i in range(10):
#     print(f"{i}: {counter_dict[i]/total*100}")

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,64)
        self.fc4=nn.Linear(64,10)
    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return F.log_softmax(x,dim=1)

net=Net()


import torch.optim as optim
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

# print(dir(nn.Module))        

for epoch in range(3):
    for x,y in train_loader:
        net.zero_grad()
        output=net(x.view(-1,784))
        loss=F.nll_loss(output,y)
        loss.backward()
        optimizer.step()
    # print(loss)
correct=0
total=0

for x,y in train_loader:
    output=net(x.view(-1,784))
    for index,i in enumerate(output):
        if (y[index]==torch.argmax(i)):
            correct+=1
            total+=1
        else:
            total+=1
print(f"accuracy={correct/total}")

        
    
        
        