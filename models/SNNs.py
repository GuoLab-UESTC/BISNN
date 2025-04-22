
import torch
from models import func as sf
import torch.nn.functional as F
import torch.nn as nn


class EEGSNN1(nn.Module):
    def __init__(self, input_size, num_classes, thresh=1.0, tau=2.0, hete_th=0.01, hete_tau=0.01, train_th=False, train_tau=False, P=10, time_windows=32):
        super(EEGSNN1, self).__init__()
        self.time_windows = time_windows
        self.conv1 = sf.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1, bias=True)
        if train_th:
            self.lif1  = sf.LIF(thresh=torch.ones([32,self.time_windows])*thresh, tau=torch.ones([32,self.time_windows])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif1  = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.pool1 = sf.AdaptiveMaxPool1d(output_size=self.time_windows//2)
        self.conv2 = sf.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True)
        if train_th:
            self.lif2  = sf.LIF(thresh=torch.ones([64,self.time_windows//2])*thresh, tau=torch.ones([64,self.time_windows//2])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif2  = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.pool2 = sf.AdaptiveMaxPool1d(output_size=self.time_windows//4)
        self.fc1 = sf.Linear(64 * (self.time_windows // 4), 128, bias=True)
        if train_th:
            self.lif3   = sf.LIF(thresh=torch.ones([128])*thresh, tau=torch.ones([128])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif3   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.fc2 = sf.Linear(128, num_classes, bias=True)
        if train_th:
            self.lif4   = sf.LIF(thresh=torch.ones([num_classes])*thresh, tau=torch.ones([num_classes])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif4   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.boost  = nn.AvgPool1d(P, P)

    def forward(self,input):
        out = input.reshape(input.shape[0],int(input.shape[1]/self.time_windows),self.time_windows,input.shape[-1])
        out = self.conv1(out.permute(0,1,3,2))
        out, _ = self.lif1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out, _ = self.lif2(out)
        out = self.pool2(out)
        out = out.view(-1,input.shape[1]//self.time_windows,64*(self.time_windows//4))
        out = self.fc1(out)
        out, _  = self.lif3(out)
        out = self.fc2(out)
        _, out  = self.lif4(out)
        outputs = self.boost(out).mean(1)
        return outputs


class EEGSNN1_(nn.Module):
    def __init__(self, input_size, num_classes, thresh=1.0, tau=2.0, hete_th=0.01, hete_tau=0.01, train_th=False, train_tau=False, P=10, time_windows=32):
        super(EEGSNN1_, self).__init__()
        self.time_windows = time_windows
        self.conv1 = sf.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(128*3//time_windows)
        if train_th:
            self.lif1  = sf.LIF(thresh=torch.ones([32,64])*thresh, tau=torch.ones([32,64])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif1  = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.pool1 = sf.AdaptiveMaxPool1d(output_size=self.time_windows//2)
        self.conv2 = sf.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.batchnorm2 = nn.BatchNorm2d(128*3//time_windows)
        if train_th:
            self.lif2  = sf.LIF(thresh=torch.ones([64,32])*thresh, tau=torch.ones([64,32])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif2  = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.pool2 = sf.AdaptiveMaxPool1d(output_size=self.time_windows//4)
        self.fc1 = sf.Linear(64 * (self.time_windows // 4), 128, bias=True)
        if train_th:
            self.lif3   = sf.LIF(thresh=torch.ones([128])*thresh, tau=torch.ones([128])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif3   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.fc2 = sf.Linear(128, num_classes, bias=True)
        if train_th:
            self.lif4   = sf.LIF(thresh=torch.ones([num_classes])*thresh, tau=torch.ones([num_classes])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif4   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.boost  = nn.AvgPool1d(P, P)

    def forward(self,input):
        out = input.reshape(input.shape[0],int(input.shape[1]/self.time_windows),self.time_windows,input.shape[-1])
        out = self.conv1(out.permute(0,1,3,2))
        out = self.batchnorm1(out)
        out, _ = self.lif1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out, _ = self.lif2(out)
        out = self.pool2(out)
        out = out.view(-1,input.shape[1]//self.time_windows,64*(self.time_windows//4))
        out = self.fc1(out)
        out, _  = self.lif3(out)
        out = self.fc2(out)
        _, out  = self.lif4(out)
        outputs = self.boost(out).mean(1)
        return outputs



class EEGSNN2(nn.Module):
    def __init__(self, input_size, num_classes, thresh=1.0, tau=2.0, hete_th=0.01, hete_tau=0.01, train_th=False, train_tau=False, P=10, time_windows=32):
        super(EEGSNN2, self).__init__()
        self.time_windows = time_windows
        self.layer1 = sf.Linear(input_size*self.time_windows, 512, bias=True)
        if train_th:
            self.lif1   = sf.LIF(thresh=torch.ones([512])*thresh, tau=torch.ones([512])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        else:
            self.lif1   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.layer2 = sf.Linear(512, num_classes, bias=True)
        if train_th:
            ## self.lif2   = sf.LIF(thresh=torch.ones([num_classes*P])*thresh, tau=torch.ones([num_classes*P])*tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
            self.lif2   = sf.LIF(thresh=thresh, tau=tau, hete_th=0.0, hete_tau=0.0, train_th=False, train_tau=False)
        else:
            self.lif2   = sf.LIF(thresh=thresh, tau=tau, hete_th=hete_th, hete_tau=hete_tau, train_th=train_th, train_tau=train_tau)
        self.boost  = nn.AvgPool1d(P, P)

    def forward(self,input):
        out = input.reshape(input.shape[0],int(input.shape[1]/self.time_windows),self.time_windows,input.shape[-1])
        out = self.layer1(out.flatten(2,3).contiguous())
        out, _ = self.lif1(out)
        out = self.layer2(out)
        out, _ = self.lif2(out)
        outputs = self.boost(out).mean(1)
        return outputs
