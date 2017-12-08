from CNN_model import ResBlock,ResNet
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = dsets.CIFAR10(
    root='./data', train=True, transform=train_transform, download=True)

test_dataset = dsets.CIFAR10(
    root='./data', train=False, transform=test_transform)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

resnet = ResNet(ResBlock, [3, 3, 3])
resnet.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

# Training
total_epoch = 50
for epoch in range(total_epoch):
    running_loss = 0
    running_acc = 0
    running_num = 0
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # =====================log=====================
        running_num += labels.size(0)
        running_loss += loss.data[0] * labels.size(0)
        _, correct_label = torch.max(outputs, 1)
        correct_num = (correct_label == labels).sum()
        running_acc += correct_num.data[0]
        if (i + 1) % 100 == 0:
            print_loss = running_loss / running_num
            print_acc = running_acc / running_num
            print("Epoch [{}/{}], Iter [{}/{}] Loss: {:.6f} Acc: {:.6f}".
                  format(epoch + 1, total_epoch, i + 1,
                         len(train_loader), print_loss, print_acc))

    # Decaying Learning Rate
    if (epoch + 1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

# Test
correct = 0
total = 0
for images, labels in test_loader:
    if torch.cuda.is_available:
        images = Variable(images.cuda())
    else:
        images = Variable(images)
    outputs = resnet(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the model on the test images: {:.2f} %%'.format(
    100 * correct / total))

# Save the Model
torch.save(resnet.state_dict(), 'resnet.pth')
