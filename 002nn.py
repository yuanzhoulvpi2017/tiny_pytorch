import torch.nn as nn 
import torch.nn.functional as F 
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


def init_weights(m):
    print(m)
    print('*' * 10)
    if type(m) == nn.Linear:
        m.weight.data.fill_(1.0)
        print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)


for buf in net.buffers():
    print(type(buf.data), buf.size())

l = nn.Linear(2, 2)
net = nn.Sequential(l, l)
for idx, m in enumerate(net.modules()):
    print(idx, '->', m)

linear = nn.Linear(2, 2)
linear.weight

linear.to(torch.double)
linear.weight

gpu0 = torch.device('cuda:0')
linear.to(gpu0, dtype=torch.half, non_blocking=True)
linear.weight
cpu = torch.device('cpu')
linear.to(cpu)
linear.weight

#example of using sequential
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)
#example of suing sequential with OrderedDict
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 64, 5)),
    ('relu2', nn.ReLU())
]))

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        #ModuleList can act an iterable , or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
            'conv', nn.Conv2d(10, 10, 3),
            'pool', nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU()],
            ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ParameterList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x

m = nn.Conv1d(16, 33, 3, stride=2)
input1 = torch.randn(20, 16, 50)
output1 = m(input1)
output1.size()
type(output1)
