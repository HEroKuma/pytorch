import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y):
    plt.plot(x, y, 'o')
    plt.title("training data")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()
# to instantiate a model, create a class inherit from nn.Module
class LinearRegression(nn.Module):
    def __init__(self):
        # clair the modules, ex:
        # self.conv1 = nn.Conv2d(1,3,3)
        # self.pool1 = nn.MaxPool2d(2,2)
        # self.fc1 = nn.Linear(10,10)
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    # use forward function to defines the network structure
    # the structure allow you that can use mult input
    def forward(self, input):
        out = self.linear(input)
        return out

x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.042],
                    [10.791],[5.313],[7.997],[3.1]], dtype=np.float32)

y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],
                    [3.336],[2.596],[2.53],[1.221],[2.827],
                    [3.465],[1.65],[2.904],[1.3]], dtype=np.float32)

#plot_data(x_train, y_train)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    inputs = Variable(x_train).cuda()
    target = Variable(y_train).cuda()

    out = model(inputs)
    # compare the output and target's distance
    loss = criterion(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%100 is 0:
        print('epoch[{}/{}], loss: {:.6f}'.format(epoch+1, num_epochs, loss.data[0]))

model.eval()
predict = model(Variable(x_train).cuda())
predict = predict.cpu()
predict = predict.data.numpy()
print(predict)
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting line')
plt.show()