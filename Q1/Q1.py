#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
def sigmoid(n):
    result = 1 / (1 + np.exp(-n))
    return result

z = np.array([[5,-1],[2,-1],[3.1,-1],])          #Input
print("Input:\n",z)
print("The dimension of input",z.shape)
z1 = np.array([[5,1,-1],[2,1,-1],[3.1,1,-1],])

d = np.array([[0],[1],[1],[0],]) 
print("Teachers value:\n",d)                            #Teacher values

print("Weights V :\n")
v = np.array([[1,2,3],[2,6,5],])                        #Initiate v matrix
print("Initial Value of v:\n",v)

print("Weights W:\n")
w1= np.array([1,7,-1],)                                 #Initiate weight w
w1=w1.reshape(3,1)
print("Initial Weight:\n",w1)

#### Forward propagation  ####
neta=3.5                                                 #Fix neta value
iteration=5000                                           #set Iteration up to 5000
for i in range(1,iteration):
    for j,n in enumerate(z):
        #print("----Iteration----",i)
        y_net = np.dot(v,z1[j])                           #calculate y_net
        #print("Value of y_net:\n",y_net)
        y = sigmoid(y_net)                               #calculate y value
        y = np.append(y,[1])
        y = y.reshape(3,1)
        #print("Value of y:\n",y)
        wt = np.transpose(w)                              #Transpose weight matrix
        #print("Transpose shape:\n",wt.shape)
        out_net = np.dot(wt,y)                             #calculate out_net value
        out = sigmoid(out_net)
        #print("out value:\n",out)
        del_o =(d[j]-out)*(1-out)*out                      #calculate del_o Unipolar function
        #print("delta out:",del_o)
        del_hid = del_o*((y)*(1-y))*w                      #Hidden Layer
        #print("del_hid value:\n",del_hid)
        w=w+neta*del_o*y                                   #Update weight
        #print("weight value:\n",w)
        del_hid1=np.delete(del_hid,-1)
        del_hid1=del_hid1.reshape(2,1)
        v=v+neta*del_hid1*z1[j]                             #update v value and print
print("Final weight w",w)
print("Final V",v)


# In[ ]:




