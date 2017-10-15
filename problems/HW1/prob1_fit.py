import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
#Please note this is a script written in python 3 but should work in python 2.7
'''
IST 597: Foundations of Deep Learning
Problem 1: Univariate Regression

@author - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.02# step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 1000 # number of epochs (full passes through the dataset)

#Please note this is a script written in python 3 but should work in python 2.7
# begin simulation

# begin simulation

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
    yhat = np.dot(X,theta);
    return yhat

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
    gll =0;
    return gll
	
def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
    m = X.shape[0]
    yhat = regress(X,theta);
    cost =sum(pow(yhat-y,2))/(2*m);
    return cost

def computeGrad(X, y, theta): 
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    yhat = regress(X,theta);
    m = X.shape[0]
    dL_db = (np.dot(np.transpose(yhat-y),X[:,0]))/m ;
    dL_dw = (np.dot(np.transpose(yhat-y),X[:,1:m-1]))/m;
    grad = (np.dot(np.transpose(yhat-y),X))/m;
    nabla = (dL_db, dL_dw) ;# nabla represents the full gradient
    return nabla
path = os.getcwd() + '/data/prob1.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# display some information about the dataset itself here
# WRITEME: write your code here to print out information/statistics about the data-set "data" using Pandas (consult the Pandas documentation to learn how)
data.head()
data.describe()
# WRITEME: write your code here to create a simple scatterplot of the dataset itself and print/save to disk the result
data.plot(kind='scatter',x='X', y='Y', s=25,figsize=(8,6))
plt.title('Food Truck Dataset',fontsize = 15)
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.savefig('prob1\plot_DataSet.png')


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)


# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = [b, w]
Theta = np.asarray(theta)

#Adding a bais vector to our dataset
bais = np.ones([data.shape[0],1])
X_1 = np.append(bais,X,axis=1)

L = computeCost(X_1, y, Theta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = [] # you can use this list variable to help you create the loss versus epoch plot at the end (if you want)

cost.append(L)# you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
error = 99999

while( abs(error) > eps and i < n_epoch  ):
    grad = computeGrad(X_1, y,Theta)
    Theta[0]=Theta[0] -alpha*grad[0];
    Theta[1]=Theta[1] -alpha*grad[1];
    # (note: don't forget to override the theta variable...)
    L = computeCost(X_1, y, Theta) # track our loss after performing a single step
    cost.append(L) 
    error = cost[i]- cost[i+1]
    if i % 10 ==0:
        print(" {0} L = {1}".format(i,L))
    i = i +1
    
    
# print parameter values found after the search
print("w = ",Theta[1])
print("b = ",Theta[0])
print("n_epoch = ",i)
print("Training Error =",cost[-1])

kludge = 0.9 # helps with printing the plots (you can tweak this value if you like)
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)
bais_1 = np.ones([X_test.shape[0],1])
X_test1 = np.append(bais_1,X_test,axis=1)
plt.figure(figsize=(8,6))
plt.plot(X_test, regress(X_test1, Theta), label="Model")
plt.scatter(X_1[:,1], y, edgecolor='g', s=25, label="Samples")
plt.xlabel("X",fontsize = 15)
plt.ylabel("Y",fontsize=15)
plt.title(r'Linear Regression Model with $\alpha = %s$'%(alpha),fontsize =15)
plt.xlim((np.amin(X_test1) - kludge, np.amax(X_test1) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="upper left",fontsize = 15)
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.savefig('prob1\plot_Fitted_Model_alpha_%s.png'%(alpha))
plt.show() # convenience command to force plots to pop up on desktop
# visualize the loss as a function of passes through the dataset
# WRITEME: write your code here create and save a plot of loss versus epoch

plt.figure(figsize=(8,6))
plt.plot(cost[10:])

plt.xlabel("Epoch",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.title(r'Linear Regression Model with $\alpha = %s$'%(alpha),fontsize =15)
plt.savefig('prob1\plot_Loss_alpha_%s.png'%(alpha))
plt.show() # convenience command to force plots to pop up on desktop
#Please note this is a script written in python 3 but should work in python 2.7
