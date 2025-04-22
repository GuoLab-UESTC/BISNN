import torch
import torch.nn.functional as F
import torch.nn as nn


## for CNN architectures
class output_kl(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(output_kl, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t,reduction='batchmean') * (self.T**2)
        return loss


## for CNN architectures
class feature_kl(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, fun='mse', T=20):
        super(feature_kl, self).__init__()
        self.fun = fun
        self.T = T

    def forward(self, feature_stu, feature_tea):
        loss_all = 0
        for i in range(len(feature_stu)):
            if self.fun == 'mse':
                loss_all += F.mse_loss(feature_stu[i], feature_tea[i].detach())
            elif self.fun == 'l1':
                loss_all += F.l1_loss(feature_stu[i],feature_tea[i].detach())
            elif self.fun == 'kl':
                loss_all += KL_loss(feature_stu[i], feature_tea[i].detach(), self.T)
            elif self.fun == 'norm':
                loss_all = Norm_fun(feature_stu,feature_tea)
                return loss_all
            elif self.fun == "mag":
                loss_all += F.l1_loss(torch.norm(feature_stu[i]),torch.norm(feature_tea[i]))
        return loss_all/len(feature_stu)


## for ViT architectures
class output_mse(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=None):
        super(output_mse, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        if self.T == None:
            loss = F.mse_loss(y_s, y_t.detach())
            return loss
        else:
            p_s = F.log_softmax(y_s/self.T, dim=1)
            p_t = F.softmax(y_t/self.T, dim=1)
            loss = F.cross_entropy(p_s, p_t.detach()) * (self.T**2)
            # loss = F.mse_loss(p_s, p_t.detach()) * (self.T**2)
            return loss

