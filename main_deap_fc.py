import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from preprocessing import dataset_prepare
from models import func as sf
import torch.nn.functional as F
from models import ANNs, SNNs
from loss_funs import output_kl
import random
import os
import numpy as np


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_all(2024-1)
batch_size = 200
epochs =100
#preprocess, label_type[0] is 0 for valence 1 for arousal, label_type[1] represent the number of classes in the task
train_set, test_set = dataset_prepare(label_type = [0,3])
train_loader = data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path_save = '/home/sunhongze/PyProjects/IJCNN/checkpoints/'
path_save = '/home/sunhongze/PyProject/ANN-SNN/IJCNN/checkpoints/'
is_bias=True
model_name = 'fc'

def test(model):
    test_acc = 0.
    sum_sample = 0.
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            test_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()
    return test_acc.data.cpu().numpy()/sum_sample

def train(model, epochs, optimizer, scheduler=None, name=None):
    acc_list = []
    best_acc = 0
    path = path_save+name+'/'
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()
            predictions = model(images)
            _, predicted = torch.max(predictions.data, 1)
            train_loss = criterion(predictions,labels)
            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()
        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc= test(model)
        acc_list.append(train_acc)
        if valid_acc>best_acc and train_acc>0.80:
            best_acc = valid_acc
            torch.save(model, path+model_name+str(best_acc)[:7]+'-deap.pth')
        print('Epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch, train_loss_sum/len(train_loader), train_acc,valid_acc), flush=True)
    return best_acc

def train_with_kl(model_tea, model_stu, epochs, optimizer, scheduler=None, name=None):
    acc_list = []
    best_acc = 0
    path = path_save+name+'/'
    criterion_label = nn.CrossEntropyLoss().cuda()
    criterion_kl = output_kl(T=1).cuda()
    model_tea.eval()
    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0
        loss_label_sum = 0
        loss_kl_sum = 0
        model_stu.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_preds = model_tea(images)
            student_preds = model_stu(images)
            _, predicted = torch.max(student_preds.data, 1)
            loss_label = criterion_label(student_preds,labels)
            loss_kl    = criterion_kl(student_preds, teacher_preds)
            train_loss = 1.0*loss_label + 0.1*loss_kl
            train_loss.backward()
            loss_label_sum += loss_label.item()
            loss_kl_sum    += loss_kl.item()
            train_loss_sum += train_loss.item()
            optimizer.step()
            labels = labels.cpu()
            predicted = predicted.cpu().t()
            train_acc += (predicted ==labels).sum()
            sum_sample+=predicted.numel()
        if scheduler:
            scheduler.step()
        train_acc = train_acc.data.cpu().numpy()/sum_sample
        valid_acc= test(model_stu)
        acc_list.append(train_acc)
        if valid_acc>best_acc and train_acc>0.80:
            best_acc = valid_acc
            torch.save(model_stu, path+model_name+str(best_acc)[:7]+'-deap.pth')
        print('Epoch: {:3d}, Train Loss: {:.4f}, Loss label: {:.4f}, Loss kl: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f}'.format(epoch, train_loss_sum/len(train_loader), loss_label_sum/len(train_loader), loss_kl_sum/len(train_loader), train_acc,valid_acc), flush=True)
    return best_acc


print('======================================= ANN =======================================')
model_ANN = ANNs.EEGANN2(input_size=32, num_classes=3, time_windows=64).to(device)
n_parameters_ANN = sum(p.numel() for p in model_ANN.parameters() if p.requires_grad)
print(f"number of params: {n_parameters_ANN}")
optimizer_ann = torch.optim.Adam(model_ANN.parameters(), lr=0.001)
scheduler_ann = CosineAnnealingLR(optimizer_ann, epochs)
acc_ann = train(model_ANN, epochs, optimizer_ann, scheduler_ann, name='ANN')


print('======================================= SNN =======================================')
model_SNN = SNNs.EEGSNN2(input_size=32, num_classes=3, thresh=1.0, tau=2.0, hete_th=0.0, hete_tau=0.0, train_th=False, train_tau=False, P=1, time_windows=64).to(device)
n_parameters_SNN = sum(p.numel() for p in model_ANN.parameters() if p.requires_grad)
print(f"number of params: {n_parameters_SNN}")
optimizer_snn = torch.optim.Adam(model_SNN.parameters(), lr=0.001)
scheduler_snn = CosineAnnealingLR(optimizer_snn, epochs)
acc_snn = train(model_SNN, epochs, optimizer_snn, scheduler_snn, name='SNN')
del model_SNN

print('======================================= BSNN =======================================')
model_BSNN = SNNs.EEGSNN2(input_size=32, num_classes=3, thresh=1.0, tau=2.0, hete_th=0.01, hete_tau=0.01, train_th=True, train_tau=True, P=1, time_windows=64).to(device)
n_parameters_BANN = sum(p.numel() for p in model_BSNN.parameters() if p.requires_grad)
print(f"number of params: {n_parameters_BANN}")
thresh_params = []
rest_params = []
for pname, p in model_BSNN.named_parameters():
    if pname[-6:]=='thresh':
        thresh_params += [p]
    else:
        rest_params += [p]
optimizer_bsnn = torch.optim.Adam([{'params':thresh_params,'lr':0.001*0.1},{'params':rest_params,'lr':0.001}], weight_decay=0)
scheduler_bsnn = CosineAnnealingLR(optimizer_bsnn, epochs)
acc_bsnn = train(model_BSNN, epochs, optimizer_bsnn, scheduler_bsnn, name='BSNN')
del model_BSNN

print('======================================= ISNN =======================================')
model_ISNN = SNNs.EEGSNN2(input_size=32, num_classes=3, thresh=1.0, tau=2.0, hete_th=0.0, hete_tau=0.0, train_th=False, train_tau=False, P=1, time_windows=64).to(device)
n_parameters_ISNN = sum(p.numel() for p in model_ISNN.parameters() if p.requires_grad)
print(f"number of params: {n_parameters_ISNN}")
optimizer_isnn = torch.optim.Adam(model_ISNN.parameters(), lr=0.001)
scheduler_isnn = CosineAnnealingLR(optimizer_isnn, epochs)
model_dict =  model_ISNN.state_dict()
state_dict = {k:v for k,v in model_ANN.state_dict().items() if k in model_dict.keys()}
model_dict.update(state_dict)
model_ISNN.load_state_dict(model_dict)
model_ISNN.to(device)
acc_isnn = train_with_kl(model_ANN, model_ISNN, epochs, optimizer_isnn, scheduler_isnn, name='ISNN')
del model_ISNN

print('=================================== BISNN ===================================')
model_BISNN = SNNs.EEGSNN2(input_size=32, num_classes=3, thresh=1.0, tau=2.0, hete_th=0.01, hete_tau=0.01, train_th=True, train_tau=True, P=1, time_windows=64).to(device)
n_parameters_BISNN = sum(p.numel() for p in model_BISNN.parameters() if p.requires_grad)
print(f"number of params: {n_parameters_BISNN}")
thresh_params = []
rest_params = []
for pname, p in model_BISNN.named_parameters():
    if pname[-6:]=='thresh':
        thresh_params += [p]
    else:
        rest_params += [p]
optimizer_bisnn = torch.optim.Adam([{'params':thresh_params,'lr':0.001*0.1},{'params':rest_params,'lr':0.001}], weight_decay=0)
scheduler_bisnn = CosineAnnealingLR(optimizer_bisnn, epochs)
model_dict =  model_BISNN.state_dict()
state_dict = {k:v for k,v in model_ANN.state_dict().items() if k in model_dict.keys()}
model_dict.update(state_dict)
model_BISNN.load_state_dict(model_dict)
model_BISNN.to(device)
acc_bisnn = train_with_kl(model_ANN, model_BISNN, epochs, optimizer_bisnn, scheduler_bisnn, name='BISNN')


print('======================================================================')
print('Best acc of ANN is   ', acc_ann)
print('Best acc of SNN is   ', acc_snn)
print('Best acc of BSNN is  ', acc_bsnn)
print('Best acc of ISNN is  ', acc_isnn)
print('Best acc of BISNN is ', acc_bisnn)
