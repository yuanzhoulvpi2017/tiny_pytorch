import torch
from torch import nn, optim
import torch.nn.functional as F 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import time

#
batch_size = 32
learning_rate = 1e-3
num_epochs = 100

train_dataset = datasets.MNIST(
    root='D:/data', train=True, transform=transforms.ToTensor(), 
    download=True 
)
test_dataset = datasets.MNIST(
    root='D:/data', train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Logistic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)
    
    def forward(self, x):
        out = self.logistic(x)
        return out

model = Logistic_Regression(28 * 28, 10)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('*' * 10)
    print("epoch {}".format(epoch + 1))
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        
        #向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data.item()
        #向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print("[{} / {}] Loss: {:.6f}, Acc: {:.6f}".format(
                epoch + 1, num_epochs, running_loss / (batch_size * i),
                running_acc / (batch_size * i)
            ))
    print("Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}".format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))   
        ))
    model.eval()
    eval_loss = 0.
    eval_acc = 0. 
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            with torch.no_grad():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
        else:
            with torch.no_grad():
                img = Variable(img)
                label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
    print('Time:{:.1f} s'.format(time.time() - since))
    print()

torch.save(model.state_dict(), './logstic.pth')

model.state_dict().keys()
model.state_dict()['logistic.weight'].size()
