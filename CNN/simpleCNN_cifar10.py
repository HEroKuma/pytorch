import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

'''
1. Loading and normalizing CIFAR10
Using torchvision, it’s extremely easy to load CIFAR10.
'''

# ToTensor:The output of torchvision datasets are PILImage images of range [0, 1]. 
# Normalize:We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [ transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                          shuffle=True, num_workers=2)

test_data = torchvision.datasets.CIFAR10(root='./CIFAR10data', train=False,
                                       download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog','frog','horse','ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# 2. Define a Convolution Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net()

# 3. Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimzer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9) # SGD(传入参数，定义lr,动量）

# 4. Train the network
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):  
        input, target = data
        input, target = Variable(input),Variable(target)
        optimzer.zero_grad()
        output = net(input)
        loss = criterion(output,target)
        loss.backward()
        optimzer.step()
        running_loss += loss.data[0]
        if i % 2000 ==1999:   # print every 2000 mini_batches,1999,because of index from 0 on
            print ('[%d,%5d]loss:%.3f' % (epoch+1,i+1,running_loss/2000))
            running_loss = 0.0
print('Finished Training')

# 5. Test the network on the test data
dataiter = iter(test_loader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth:',' '.join('%5s' % classes[labels[j]] for j in range(4)))
outputs = net(Variable(images))
_, pred = torch.max(outputs.data,1)
print('Predicted: ', ' '.join('%5s' % classes[pred[j][0]] for j in range(4)))

correct = 0.0
total = 0
for data in test_loader:
    images,labels = data
    outputs = net(Variable(images))
    _, pred = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (pred == labels).sum()
print('Accuracy of the network on the 10000 test images : %d %%' % (100 * correct / total))

# 6. what are the classes that performed well, and the classes that did not perform well:
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in test_loader:
    images, labels = data
    outputs = net(Variable(images))
    _, pred = torch.max(outputs.data,1)
    c = (pred == labels).squeeze() # 1*10000*10-->10*10000
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' %(classes[i],100 * class_correct[i]/class_total[i]))
