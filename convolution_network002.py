import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

batch_size = 128
learning_rate = 1e-2
num_epoches = 5


def to_np(x):
    return x.cpu().data.numpy()


train_dataset = datasets.MNIST(
    root='D:/data', train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = datasets.MNIST(
    root='D:/data', train=False,
    transform=transforms.ToTensor()
)

trian_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120), 
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )
    
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = Cnn(1, 10)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epoches):
    #print("epoch {}".format(epoch + 1))
    #print("*" * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(trian_loader, 1):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        #向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.data.item()
        #后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print("[{} / {}] Loss: {:.6f}, Acc: {:.6f}".format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)
            ))
    print("Finish {} epoch, Loss: {:.6f}, ACC: {:.6F}".format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))
    ))
    #model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img_t, label_t = data
        if use_gpu:
            with torch.no_grad():
                img_t = Variable(img_t).cuda()
                label_t = Variable(label_t).cuda()

        out = model(img_t)
        loss = criterion(out, label_t)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label_t).sum()
        eval_acc += num_correct.data.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))



torch.save(model.state_dict(), './cnn.pth')
label.size()