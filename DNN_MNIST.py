import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DNN(nn.Module):
    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden1)
        self.layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer3 = nn.Linear(n_hidden2, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

class Batch_DNN(nn.Module):
    def __init__(self, in_dim, n_hidden1, n_hidden2, out_dim):
        super(Batch_DNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden1))
        self.layer1_bn = nn.BatchNorm1d(n_hidden1)
        self.layer2 = nn.Sequential(nn.Linear(n_hidden1, n_hidden2))
        self.layer2_bn = nn.BatchNorm1d(n_hidden2)
        self.layer3 = nn.Linear(n_hidden2, out_dim)

    def forward(self, x):
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = F.relu(self.layer2_bn(self.layer2(x)))
        x = F.relu(self.layer3(x))
        return x

batch_size = 64
learning_rate = 1e-2
num_epoches = 20

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(
    root='./data', train=True, transform=data_tf, download=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model = DNN(28 * 28, 300, 100, 10)
model = Batch_DNN(28 * 28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    print('epoch {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                i,
                len(train_loader), running_loss / (batch_size * i), running_acc
                / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()  # for the testing set, we don't need forward
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data[0]
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)),
                                              eval_acc / (len(test_dataset))))
print('save model ...')

torch.save(model.state_dict(), './neural_network.pth')
print('model saved!')