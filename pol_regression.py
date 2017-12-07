import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def make_features(x):
    # create matrix with col [1, x, x^2, x^3]
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 3)], 1)

# target function is y = b + w1*x + w2*x^2 + w3*x^3
class PolRegression(nn.Module):
    def __init__(self):
        super(PolRegression, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

x = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.042],
                    [10.791],[5.313],[7.997],[3.1]], dtype=np.float32)

y = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],
                    [3.336],[2.596],[2.53],[1.221],[2.827],
                    [3.465],[1.65],[2.904],[1.3]], dtype=np.float32)


#plot_data(x_train, y_train)

x_train = torch.from_numpy(x)
#print(x_train.numpy())
y_train = torch.from_numpy(y)
if torch.cuda.is_available():
    model = PolRegression().cuda()
else:
    model = PolRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

num_epochs = 10000
for epoch in range(num_epochs):
    inputs = Variable(x_train).cuda()
    target = Variable(y_train).cuda()

    out = model(inputs)
    # compare the output and target's distance
    loss = criterion(out, target)
    print("loss={}".format(loss.data[0]))
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

model.eval()
a = torch.arange(0, 15)
a.resize_(15,1)
predict = model(Variable(a).cuda())
predict = predict.cpu()
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(np.arange(0,15,1), predict, label='Fitting line')
plt.show()