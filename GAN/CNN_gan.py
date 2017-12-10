import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if not os.path.exists('./cnngan_img'):
    os.mkdir('./cnngan_img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 128
num_epoch = 100
z_dimension = 100

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mnist = datasets.MNIST(root='../data/',
                       train=True,
                       transform=img_transform,
                       download=True)

dataloader = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=batch_size,
                                         shuffle=True)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.layer2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.layer3 = nn.Linear(64*7*7, 1024)
        self.layer4 = nn.Linear(1024, 1)

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), 0.2, inplace=True)
        out = self.pool1(out)
        out = F.leaky_relu(self.layer2(out), 0.2, inplace=True)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.layer3(out))
        out = F.sigmoid(self.layer4(out))

        return out

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.layer1 = nn.Linear(input_size, num_feature)
        self.layer2 = nn.BatchNorm2d(1)
        self.layer3 = nn.Conv2d(1, 50, 3, stride=1, padding=1)
        self.layer4 = nn.BatchNorm2d(50)
        self.layer5 = nn.Conv2d(50, 25, 3, stride=1, padding=1)
        self.layer6 = nn.BatchNorm2d(25)
        self.layer7 = nn.Conv2d(25, 1, 2, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), 1, 56, 56)
        out = F.relu(self.layer2(out), inplace=True)
        out = self.layer3(out)
        out = F.relu(self.layer4(out), inplace=True)
        out = self.layer5(out)
        out = F.relu(self.layer6(out), inplace=True)
        out = F.tanh(self.layer7(out))

        return out

D = discriminator().cuda()  # discriminator model
G = generator(z_dimension, 3136).cuda()  # generator model

criterion = nn.BCELoss()  # binary cross entropy

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# train
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)

        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()


        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                          real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './cnngan_img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './cnngan_img/fake_images-{}.png'.format(epoch+1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')