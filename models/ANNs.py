import torch
from models import func as sf
import torch.nn.functional as F
import torch.nn as nn


class EEGANN1(nn.Module):
    def __init__(self, input_size, num_classes, time_windows=32):
        super(EEGANN1, self).__init__()
        self.time_windows = time_windows
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (self.time_windows // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.reshape(x.shape[0],int(x.shape[1]/self.time_windows),self.time_windows,x.shape[-1])
        out = self.pool(torch.relu(self.conv1(out.flatten(0,1).contiguous().permute(0,2,1))))
        out = self.pool(torch.relu(self.conv2(out)))
        out = out.view(-1, 64 * (self.time_windows // 4))
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view([x.shape[0], x.shape[1] // self.time_windows, out.shape[-1]]).mean(1)


class EEGANN1_(nn.Module):
    def __init__(self, input_size, num_classes, time_windows=32):
        super(EEGANN1_, self).__init__()
        self.time_windows = time_windows
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (self.time_windows // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = x.reshape(x.shape[0],int(x.shape[1]/self.time_windows),self.time_windows,x.shape[-1])
        out = self.pool(torch.relu(self.conv1(out.flatten(0,1).contiguous().permute(0,2,1))))
        out = self.pool(torch.relu(self.conv2(out)))
        out = out.view(-1, 64 * (self.time_windows // 4))
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view([x.shape[0], x.shape[1] // self.time_windows, out.shape[-1]]).mean(1)


class EEGANN2(nn.Module):
    def __init__(self, input_size, num_classes, time_windows=32):
        super(EEGANN2, self).__init__()
        self.time_windows = time_windows
        self.layer1 = nn.Linear(input_size*self.time_windows,512)
        self.layer2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = x.reshape(x.shape[0],int(x.shape[1]/self.time_windows),self.time_windows,x.shape[-1])
        out = F.relu6(self.layer1(out.flatten(0,1).flatten(1,2).contiguous()))/1.0
        out = F.relu6(self.layer2(out))/1.0
        return out.view([x.shape[0], int(x.shape[1]/self.time_windows), out.shape[-1]]).mean(1)
