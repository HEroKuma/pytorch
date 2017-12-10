import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

if not os.path.exists('./gan_img'):
    os.mkdir('./gan_img')

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
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])

mnist = datasets.MNIST(root='../data/',
                       train=True,
                       transform=img_transform,
                       download=True)
# Data loader
dataloader = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=batch_size,
                                         shuffle=True)
class discrimintor(nn.Module):
    def __init__(self):
        super(discrimintor, self).__init__()
        self.layer1 = nn.Linear(28*28, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.leaky_relu(self.layer1(x), 0.2)
        out = F.leaky_relu(self.layer2(out), 0.2)
        out = F.sigmoid(self.layer3(out))

        return out

class generater(nn.Module):
    def __init__(self):
        super(generater, self).__init__()
        self.layer1 = nn.Linear(100, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 784)

    def forward(self, x):
        out = F.relu(self.layer1(x), True)
        out = F.relu(self.layer2(out), True)
        out = F.tanh(self.layer3(out))

        return out


D = discrimintor()
G = generater()
if torch.cuda.is_available():
    D = D.cuda()
G = G.cuda()

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # train discriminator
        img = img.view(num_img, -1)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # loss of real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out

        # loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'
                  .format(epoch, num_epoch, d_loss.data[0], g_loss.data[0],
                          real_scores.data.mean(), fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './gan_img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './gan_img/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')