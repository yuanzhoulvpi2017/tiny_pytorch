import torch
import torchvision
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os 

if not os.path.exists("D:/data/dc_img"):
    os.mkdir("D:/data/dc_img")

def to_img(x):
    x = 0.5 *(x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 10
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('D:/data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = Variable(img).cuda()
        #forward
        output = model(img)
        loss = criterion(output, img)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch [{} / {}], Loss: {:.4f}".format(
        epoch+1, num_epochs, loss.data.item()
    ))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, "D:/data/dc_img/images_{}.png".format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')