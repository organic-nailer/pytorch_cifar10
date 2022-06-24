import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms

data_path = './data/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.ToTensor())
cifar10_validation = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()

        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)

        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)])
        )
        self.fc1 = nn.Linear(8*8*n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8*8*self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
device



loaded_model = NetResDeep().to(device=device)
loaded_model.load_state_dict(torch.load(data_path + "cifar10.pt"))

train_loader = torch.utils.data.DataLoader(cifar10, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(cifar10_validation, batch_size=64, shuffle=False)

def validate(model, train_loader, validate_loader):
    for name, loader in [("train", train_loader), ("val", validate_loader)]:
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels
                outputs = model(imgs).cpu()
                predicted = np.argmax(outputs.numpy(), axis=1)
                total += labels.shape[0]

                correct += int((predicted == labels.numpy()).sum())

            print(f"{name} acc: {correct / total:.3f}")

validate(loaded_model, train_loader, validate_loader)
