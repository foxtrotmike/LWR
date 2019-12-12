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


#from example import Example
#from classifiers import linclass_rej
#import numpy as np
from random import shuffle
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.kernel_approximation import RBFSampler
#import matplotlib.pyplot as plt #importing plotting module
from plotit import plotit




def loss_rej(y, h, r=1, c=0):
    if type(r)!=type(1):
        r=np.array(r)
    l_total=np.mean(np.sum(((y*h)<=0)*(r>0)+c*(r<=0)))
    l_rejection=np.mean(np.sum(c*(r<=0)))
    return l_total, l_rejection

def hinge(y_true, y_pred):
    zero = torch.Tensor([0]) 
#    import pdb; pdb.set_trace()
#    return torch.mean(torch.max(zero, 1 - y_true * y_pred))
    return torch.max(zero, 1 - y_true * y_pred)


class Net(nn.Module):
    def __init__(self,d):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(d,10)  
#        self.hidden2 = nn.Linear(10,10)  
        self.out = nn.Linear(10,1)

    def forward(self,x):
#        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.tanh(x)
#        x = self.hidden2(x)
#        x = F.tanh(x)
        
        x = self.out(x)
#        x = F.tanh(x)
        return x



class RejNet(nn.Module):
    def __init__(self,d):
        super(RejNet, self).__init__()
        self.hidden1 = nn.Linear(d,10)  
        self.hidden2 = nn.Linear(10,100)  
        self.out = nn.Linear(100,1)

    def forward(self,x):
#        x = x.view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.tanh(x)
        x = self.hidden2(x)
        x = F.tanh(x)
        
        x = self.out(x)
        x = F.tanh(x)
        return x



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
#Ytr=Ytr[:,None]
############## Train classifier#########################
#criterion1=nn.MSELoss#hinge
class_epochs=50000
mlp_class=Net(Xtr.shape[1])
optimizer = optim.Adam(mlp_class.parameters())

zero=np.zeros(Ytr.shape)
zero=Variable(torch.from_numpy(zero)).type(torch.FloatTensor)
#rej_epochs=2000
mlp_rej=RejNet(Xtr.shape[1])
optimizer_rej = optim.Adam(mlp_rej.parameters(), lr=0.0001)
c=0.650
#beta=1/(1-2*c)



L=[]

for epoch in range(class_epochs):
            # Forward pass: Compute predicted y by passing x to the model
    y_pred = mlp_class(Xtr)
    # Compute and print loss
#    loss=torch.sum(torch.max(torch.zeros(Ytr.shape), 1 - Ytr * y_pred))
#    loss = criterion1(y_pred, Ytr)
    #print(epoch, loss.data[0])
    # Zero gradients, perform a backward pass, and update the weights.
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
    
    
    h = torch.squeeze(mlp_class(Xtr),1)
    r=torch.squeeze(mlp_rej(Xtr),1)
    e = hinge(Ytr,h)
    #l1=torch.max(alpha/2*(r+eps_ins(Ytr,h)),c*(1.0-beta*r))
    l1 = torch.max(r+e,c*(1-r))
    #l1=torch.max(alpha/2*(r+e),c*(1.0-beta*r))

    loss_r=torch.mean(torch.max(zero, l1 ))    
#    1/0

#   
    L.append(loss_r.data.numpy())
    
#    l1=torch.max( 1.0+(r-Ytr*h)/2.0,c*(1.0-beta*r) )
#    loss_r=torch.sum(torch.max(zero, l1 ))
#    loss=loss_r
    # Compute and print loss
#    loss=torch.sum(torch.max(torch.zeros(Ytr.shape), 1 - Ytr * y_pred))
#    loss = criterion1(y_pred, Ytr)
    #print(epoch, loss.data[0])
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
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
     
################# Train Rejection model ######################
#zero=np.zeros(Ytr.shape)
#zero=Variable(torch.from_numpy(zero)).type(torch.FloatTensor)
##rej_epochs=2000
##mlp_rej=Net(Xtr.shape[1])
##optimizer_rej = optim.Adam(mlp_rej.parameters())
##c=0.4
##beta=1/(1-2*c)
##for epoch in range(class_epochs):
#            # Forward pass: Compute predicted y by passing x to the model
#    h = mlp_class(Xtr)
#    r=mlp_rej(Xtr)
#    l1=torch.max( 1.0+(r-Ytr*h)/2.0,c*(1.0-beta*r) )
#    loss_r=torch.sum(torch.max(zero, l1 ))
#    # Compute and print loss
##    loss=torch.sum(torch.max(torch.zeros(Ytr.shape), 1 - Ytr * y_pred))
##    loss = criterion1(y_pred, Ytr)
#    #print(epoch, loss.data[0])
#    # Zero gradients, perform a backward pass, and update the weights.
#    optimizer_rej.zero_grad()
#    loss_r.backward()
#    optimizer_rej.step()
#
#
#for param in mlp_rej.parameters():
#        param.requires_grad =False
#
##1/0    
#    
################# Test classifier########################
#test=examples#create_ex_gauss(m1=1.0, m2=2.0, sd=1.0, n=100)     
#X = np.array([e.features_w for e in test])
#X=Variable(torch.from_numpy(X)).type(torch.FloatTensor)
#Y = np.array([e.label for e in test])
    
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


#plt.close('all')
plt.figure()
#X2 = np.array([e.raw_features for e in test])
X2=np.array(X)
plotit(X2,Y,clf=mlp_rej, transform = None, conts =[0], ccolors = ['g'], hold = False )
plt.figure()
plotit(X2,Y,clf=mlp_class, transform = None, conts =[-1,0,1])
