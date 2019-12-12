# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:38:44 2019

@author: Amina Asif
"""

import numpy as np
import matplotlib.pyplot as plt


#import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch



from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc_roc
from sklearn import metrics
from random import shuffle
from plotit import plotit




def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
    return torch.max(zero, 1 - y_true * y_pred)


class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,10)  
        self.out = nn.Linear(10,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = F.tanh(x)     
        x = self.out(x)
        return x



class RejNet(nn.Module):
    def __init__(self,d):
        super(RejNet, self).__init__()
        self.hidden1 = nn.Linear(d,10)  
        self.hidden2 = nn.Linear(10,100)  
        self.out = nn.Linear(100,1)

    def forward(self,x):
        x = self.hidden1(x)
        x = F.tanh(x)
        x = self.hidden2(x)
        x = F.tanh(x)
        
        x = self.out(x)
        x = F.tanh(x)
        return x


######## Generate toy examples##################
d=1.50
n=500
X1=np.random.randn(n,2)+d*np.array([1,1])
l1=[1.0]*len(X1)

X2=np.random.randn(n,2)+d*np.array([1,-1])
l2=[-1.0]*len(X2)

X3=np.random.randn(n,2)+d*np.array([-1,1])
l3=[-1.0]*len(X3)


X4=np.random.randn(n,2)+d*np.array([-1,-1])
l4=[1.0]*len(X4)
data=np.vstack((X1, X2, X3, X4))
labels=np.array(l1+l2+l3+l4)

pos=np.vstack((X1,X4))
neg=np.vstack((X2,X3))
#plt.scatter(pos[:,0], pos[:,1])
#plt.scatter(neg[:,0], neg[:,1])
Xtr=Variable(torch.from_numpy(data)).type(torch.FloatTensor)

Ytr=Variable(torch.from_numpy(labels)).type(torch.FloatTensor)

############## Train classifier#########################
class_epochs=50000
mlp_class=Net(Xtr.shape[1])
optimizer = optim.Adam(mlp_class.parameters())

zero=np.zeros(Ytr.shape)
zero=Variable(torch.from_numpy(zero)).type(torch.FloatTensor)
mlp_rej=RejNet(Xtr.shape[1])
optimizer_rej = optim.Adam(mlp_rej.parameters(), lr=0.0001)
c=0.650
L=[]

for epoch in range(class_epochs):
            # Forward pass: Compute predicted y by passing x to the model
    y_pred = mlp_class(Xtr)
 
    h = torch.squeeze(mlp_class(Xtr),1)
    r=torch.squeeze(mlp_rej(Xtr),1)
    e = hinge(Ytr,h)
    l1 = torch.max(r+e,c*(1-r))
    loss_r=torch.mean(torch.max(zero, l1 ))    
#    1/0

#   
    L.append(loss_r.data.numpy())
    

    optimizer.zero_grad()
    optimizer_rej.zero_grad()
    loss_r.backward()
    optimizer.step()
    optimizer_rej.step()
    
plt.close('all')
plt.plot(L)
plt.title('Loss')
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')    
    
for param in mlp_class.parameters():
    param.requires_grad =False
        
for param in mlp_rej.parameters():
    param.requires_grad =False
     

################# Test classifier########################
   
X=Xtr
Y=Ytr

y_p= mlp_class(X)
y_p=y_p.detach()
y_r=mlp_rej(X)
y_r=y_r.detach()
y_r=y_r.numpy().flatten()
y_p2=y_p.numpy().flatten()
Y=np.array(Y)
auc_c=auc_roc(Y, y_p2)
auc_r=auc_roc(Y[y_r>0], y_p2[y_r>0])

print("AUC without rejection=", auc_c)
print("AUC with rejection=", auc_r)

print ("Number of examples rejected=", len(y_r[y_r<0]), "/", len(y_r))

plt.figure()

X2=np.array(X)
plotit(X2,Y,clf=mlp_rej, transform = None, conts =[0], ccolors = ['g'], hold = False )
plt.figure()
plotit(X2,Y,clf=mlp_class, transform = None, conts =[-1,0,1])
