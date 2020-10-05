#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import random
from scipy.io import savemat 
import os
from os import path
from sklearn.preprocessing import normalize
from librosa.util import find_files
from torch.nn.utils import clip_grad_norm_
import torch.nn.parallel.data_parallel as data_parallel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

t60s = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]

T60s = np.asarray(t60s)

class ReverbDataset(data.Dataset):
    def __init__(self, file_path):
        self.fileList = find_files(file_path,"npz")
        
    def __getitem__(self, index):
        dat = np.load(self.fileList[index])
        mix_speech = dat["mix_speech"]
        speech1 = dat["target1"]
        speech2 = dat["target2"]
        new_out = np.argmax(speech2)
       
        return torch.from_numpy(mix_speech).t().type(torch.FloatTensor), torch.tensor(speech1).type(torch.FloatTensor), torch.tensor(new_out).type(torch.LongTensor)
    def __len__(self):
        return len(self.fileList)



input_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/timit_8k_13/reverb_train/train_wlen_480_nfft_512_overlap_360/reverb_rir5000_roomNum10_t600.3_loc500_8kHz.npz'
dat = np.load(input_path)
mix_speech = dat['mix_speech']
dim = mix_speech.shape
input_feature = dim[1]

# In[ ]:
class param:
    lr = 1e-3
    num_epoches = 50
    bs = 30
    out_channel = 16
    kernel_size = 3
    pooling_size = 2
    linear_unit = 32
    in_channel = 1
    num_class = 13
    beta = 1
    
train_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/timit_8k_13/reverb_train/train_wlen_480_nfft_512_overlap_360'
train_loader = data.DataLoader(ReverbDataset(train_path), batch_size=param.bs, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

dev_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/timit_8k_13/reverb_valid/valid_wlen_480_nfft_512_overlap_360'
dev_loader = data.DataLoader(ReverbDataset(dev_path), batch_size=param.bs, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)

test_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/timit_8k_13/reverb_test2/valid_wlen_480_nfft_512_overlap_360'
test_loader = data.DataLoader(ReverbDataset(dev_path), batch_size=param.bs, shuffle=False,
                              num_workers=4, drop_last=True, pin_memory=True)


# In[6]:








# In[ ]:


class class_reverb(nn.Module):
    def __init__(self):
        super(class_reverb,self).__init__()
        self.out_channel = param.out_channel
        self.kernel_size = param.kernel_size
        self.batchSize = param.bs
        self.pooling_size = param.pooling_size
        self.in_channel = param.in_channel
        self.num_class = param.num_class
        self.linear_unit = param.linear_unit
        self.conv1_1 = nn.Conv2d(self.in_channel, self.out_channel,kernel_size = self.kernel_size)
        self.batch_norm1 = nn.BatchNorm2d(self.out_channel)
        self.conv1_2 = nn.Conv2d(self.out_channel, self.out_channel,kernel_size = self.kernel_size)
        self.max_pooling = nn.MaxPool2d(kernel_size = self.pooling_size)
        self.conv2_1 = nn.Conv2d(self.out_channel, 2* self.out_channel,kernel_size = self.kernel_size)
        self.conv2_2 = nn.Conv2d(2*self.out_channel, 2*self.out_channel,kernel_size = self.kernel_size)
        self.batch_norm2 = nn.BatchNorm2d(2*self.out_channel)
        self.conv3 = nn.Conv2d(2*self.out_channel, 4*self.out_channel,kernel_size = self.kernel_size)
        self.conv4 = nn.Conv2d(4*self.out_channel, 4*self.out_channel,kernel_size = self.kernel_size)
        self.batch_norm3 = nn.BatchNorm2d(4*self.out_channel)
        self.fc1 = nn.Linear(77760, 2*self.linear_unit)
        self.fc1_batchNorm = nn.BatchNorm1d(2*self.linear_unit)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(2*self.linear_unit, self.linear_unit)
        self.fc2_batchNorm = nn.BatchNorm1d(self.linear_unit)
        self.fc3 = nn.Linear(self.linear_unit,self.num_class)
        self.conv5 = nn.Conv2d(4*self.out_channel,8*self.out_channel,kernel_size = self.kernel_size)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc4 = nn.Linear(8*self.out_channel,self.linear_unit)
        self.fc4_batchNorm = nn.BatchNorm1d(self.linear_unit)
        self.fc5 = nn.Linear(self.linear_unit,1)
        
    def forward(self,sig):
        sig = sig.unsqueeze_(1)
        #print(sig.shape)
        out1 = Func.relu(self.conv1_1(sig))
        out1 = self.batch_norm1(out1)
        out1 = Func.relu(self.conv1_2(out1))
        out1 = self.batch_norm1(out1)
        out1 = self.max_pooling(out1)
        
        out2 = Func.relu(self.conv2_1(out1))
        out2 = self.batch_norm2(out2)
        out2 = Func.relu(self.conv2_2(out2))
        out2 = self.batch_norm2(out2)
        out2 = self.max_pooling(out2)
        
        out3 = Func.relu(self.conv3(out2))
        out3 = self.batch_norm3(out3)
        out3 = self.max_pooling(out3)
        
        out4 = Func.relu(self.conv4(out3))
        out4 = self.batch_norm3(out4)
        out4 = self.max_pooling(out4)
        
        new_in1 = out4
        new_in2 = out4
        
        new_in1 = new_in1.view(self.batchSize,-1)
        #print(new_in1.shape)
        fc_out1 = self.fc1(new_in1)
        fc_out1 = self.fc1_batchNorm(fc_out1)
        fc_out1 = self.leakyRelu(fc_out1)
        
        fc_out2 = self.fc2(fc_out1)
        fc_out2 = self.fc2_batchNorm(fc_out2)
        fc_out2 = self.leakyRelu(fc_out2)
        # Predict OneHot vector (classification)
        final2 = self.fc3(fc_out2)
       
        
        out5 = Func.relu(self.conv5(new_in2))
        #print(out5.shape)
        out5 = self.avgpooling(out5)
        #print(out5.shape)
        out5 = out5.squeeze_()
        #print(out5.shape)
        fc_out4 = self.fc4(out5)
        #print(fc_out4.shape)
        fc_out4 = self.fc4_batchNorm(fc_out4)
        fc_out4 = self.leakyRelu(fc_out4)
        # Predict T60, such as 0.3, 0.4
        final1 = Func.relu(self.fc5(fc_out4))
        final1 = final1.squeeze_()
        #print(final1)
        return final1, final2
        
        


# In[ ]:


model = class_reverb().cuda()

optim = torch.optim.Adam(model.parameters(), lr=param.lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'min',factor=0.5,patience=3,verbose=True)

# In[ ]:


criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
mae = nn.L1Loss()

# In[ ]:
def get_loss(dl, model):
    loss = 0
    correction = 0
    total = 0
    
    for X1,y1,y2 in dl:
        X1,y1,y2 = Variable(X1).cuda(),Variable(y1).cuda(),Variable(y2).cuda()
        out1,out2 = model(X1)
        out2_v = torch.argmax(out2,dim=1)
       
    
        # regression mse
        ploss1 = criterion1(out1,y1)
        # classification crossentropy
        ploss2 = criterion2(out2,y2)
       
        
        

        pLoss = param.beta * ploss1 + (1-param.beta) *ploss2 
        loss += pLoss * param.bs
        correction += out2_v.eq(y2.data).sum().item()
            #print(correction)
        total += y2.size(0)
    avgloss = loss / (len(dl.dataset))
    accuracy = correction / total
    return avgloss,accuracy

PATH = "/N/slate/liyuy/PROJECTS/DEREVERB3/model_13/model_"+ "%.1f" % param.beta + param.new_train+".pt"

#gradient_clip = 5

#model.train(True)
min_loss = np.inf
best_accuracy = 0.0
train_loss = []
val_loss = []

mse_loss = 0
correction = 0
total = 0

count = 1

mae_loss = 0

for epoch in range(1,param.num_epoches):
    loss = 0.0
    correction = 0
    total = 0
    
    model.train(True)
    with torch.set_grad_enabled(True):
        for sig,src1,src2 in train_loader:
            
            sig = Variable(sig.cuda())  
            
            src1 = Variable(src1.cuda())
            src2 = Variable(src2.cuda())
        
            
            est1,est2= model(sig)
            est2_v = torch.argmax(est2,dim=1)
  
            correction += est2_v.eq(src2.data).sum().item()
            #print(correction)
            total += src2.size(0)

            
            #MSE for regression part
            pLoss1 = criterion1(est1,src1)
            #CrossEntropy for classification part
            pLoss2 = criterion2(est2,src2)
           
           
            
            pLoss = param.beta * pLoss1 + (1-param.beta) * pLoss2 
            loss += pLoss * param.bs
            
            optim.zero_grad()
            pLoss.backward()
            #clip_grad_norm_(model.parameters(), gradient_clip)
            optim.step()
            
        avgLoss = loss/len(train_loader.dataset)
        train_accuracy = correction/total 
        #print(correction,total,train_accuracy)
    with torch.no_grad():
        model.eval()
        devLoss, devAccuracy = get_loss(dev_loader,model) 
    new_avg = avgLoss.cpu().item()
    new_dev = devLoss.cpu().item()
    train_loss.append(new_avg)
    val_loss.append(new_dev)
        
        #scheduler.step(devLoss)
    if devLoss < min_loss:
        min_loss = devLoss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': min_loss,
        }, PATH)
    if devAccuracy > best_accuracy:
        best_accuracy = devAccuracy
        
    print('Epoch {:2}, Train Loss:{:>.9f}, Train Accuracy:{:>.5f}, Validation Loss:{:>.9f},Validation Accuracy:{:>.5f}, best accuracy:{:>.5f}'.format(epoch,avgLoss,train_accuracy,devLoss,devAccuracy,best_accuracy))
    #print(epoch)
save_file = '/N/slate/liyuy/PROJECTS/DEREVERB3/results/loss/loss_'+"%.1f"%param.beta+param.new_train+'.npz'
np.savez(save_file,
        train = train_loss,
        val = val_loss) 

PATH = "/N/slate/liyuy/PROJECTS/DEREVERB3/model_13/model_"+ "%.1f" % param.beta + param.new_train+".pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
save_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/results/train_results/vectors_'+"%.1f"%param.beta+param.new_train+'.npz'


loss = 0
correction = 0
total = 0
target_list = []
est_list = []
count = 1
T60_est = []
T60_tru = []
mae_loss = 0
mse_c = 0
mae_c = 0
save_print = '/N/slate/liyuy/PROJECTS/DEREVERB3/results/train_results/vectors_'+"%.1f"%param.beta+param.new_train+'.txt'
file1 = open(save_print,'a')
with torch.no_grad():
    model.eval()
    for X1,y1,y2 in train_loader:

        X1, y1,y2 = Variable(X1).cuda(),  Variable(y1).cuda(),Variable(y2).cuda()
        #y2_n = torch.argmax(y2,dim=1)
        out1, out2 = model(X1)

        out2_v = torch.argmax(out2,dim=1)
        for i in range(0,param.bs): 
            target_list.append(y2[i].data.item())
            est_list.append(out2_v[i].data.item())
            T60_est.append(out1[i].data.item())
            T60_tru.append(y1[i].data.item())
  

        # MSE for regression
        ploss1 = criterion1(out1,y1)
       # mae for regression
        ploss3 = mae(out1,y1)
        

        # MSE for regression
        loss += ploss1.cpu().item() * param.bs
        # MAE for regression
        mae_loss += ploss3.cpu().item() * param.bs
        
        correction += out2_v.eq(y2.data).sum().item()
                #print(correction)
        total += y2.size(0)

    np.savez(save_path,
            target = target_list,
            est = est_list)    
    mseloss = loss / (len(train_loader.dataset))
    maeL1 = mae_loss / (len(train_loader.dataset))
    
    accuracy = correction / total
pcc = pearsonr(T60_tru,T60_est)
srcc = spearmanr(T60_tru,T60_est)

out = 'Train MSE Loss of regression:' + '%.9f' % mseloss + ', MAE Loss of regression:'+'%.9f' % maeL1 + ', PCC:' + '%.5f' % pcc[0] + ', SRCC:' + '%.5f' % srcc[0] + ', Accuracy:' + '%.5f' % accuracy 
file1.write(out + '\n')
print('Train MSE Loss of regression:{:>.9f}, MAE Loss of regression:{:>.9f}, PCC:{:>.5f}, SRCC:{:>.5f}, Accuracy:{:>.5f}'.format(mseloss,maeL1,pcc[0],srcc[0],accuracy))



save_path = '/N/slate/liyuy/PROJECTS/DEREVERB3/results/valid_results/vectors_'+"%.1f"%param.beta+param.new_train+'.npz'
loss = 0
correction = 0
total = 0
target_list = []
est_list = []
count = 1
T60_est = []
T60_tru = []
mae_loss = 0
mse_c = 0
mae_c = 0
#save_print = '/N/slate/liyuy/PROJECTS/DEREVERB3/results/test_results/vectors_'+"%.1f"%param.beta+param.new_train+'.txt'
#file1 = open(save_print,'a')
with torch.no_grad():
    model.eval()
    for X1,y1,y2 in dev_loader:

        X1,y1,y2 = Variable(X1).cuda(), Variable(y1).cuda(),Variable(y2).cuda()
        #y2_n = torch.argmax(y2,dim=1)
        out1, out2 = model(X1)
        
        out2_v = torch.argmax(out2,dim=1)
        for i in range(0,param.bs): 
            target_list.append(y2[i].data.item())
            est_list.append(out2_v[i].data.item())
            T60_est.append(out1[i].data.item())
            T60_tru.append(y1[i].data.item())
          

        # MSE for regression
        ploss1 = criterion1(out1,y1)
        # Cross Entropy for classification
        #ploss2 = criterion2(out2,y2)
        # MAE for regression
        ploss3 = mae(out1,y1)
       

        # MSE for regression
        loss += ploss1.cpu().item() * param.bs
        # MAE for regression
        mae_loss += ploss3.cpu().item() * param.bs
       
        correction += out2_v.eq(y2.data).sum().item()
                #print(correction)
        total += y2.size(0)

    np.savez(save_path,
            target = target_list,
            est = est_list)    
    mseloss = loss / (len(dev_loader.dataset))
    maeL1 = mae_loss / (len(dev_loader.dataset))

    accuracy = correction / total
pcc = pearsonr(T60_tru,T60_est)
srcc = spearmanr(T60_tru,T60_est)

out = 'Valid MSE Loss of regression:' + '%.9f' % mseloss + ', MAE Loss of regression:'+'%.9f' % maeL1 + ', PCC:' + '%.5f' % pcc[0] + ', SRCC:' + '%.5f' % srcc[0] + ', Accuracy:' + '%.5f' % accuracy 
file1.write(out + '\n')
print('Valid MSE Loss of regression:{:>.9f}, MAE Loss of regression:{:>.9f}, PCC:{:>.5f}, SRCC:{:>.5f}, Accuracy:{:>.5f}'.format(mseloss,maeL1,pcc[0],srcc[0],accuracy))

