from __future__ import division
import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  


#Please note this is a script written in python 3 but should work in python 2.7
'''
IST 597: Foundations of Deep Learning
Problem 3: Multivariate Regression & Classification

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
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model (LEAVE THIS FIXED TO p = 6 FOR THIS PROBLEM)
beta = 1 # regularization coefficient
alpha = 1 # step size coefficient
n_epoch =10000 # number of epochs (full passes through the dataset)
eps = 0.0000001 # controls convergence criterion
threshold = 0.5 # 
#Please note this is a script written in python 3 but should work in python 2.7

# begin simulation

# begin simulation

def sigmoid(z):
	# WRITEME: write your code here to complete the routine
    phi = 1/(1+np.exp(-z))
    return phi

def predict(X, theta):  
 	# WRITEME: write your code here to complete the routine
    oc=np.zeros(X.shape[0])
    z = regress(X,theta)
    phi = sigmoid(z)
    for i in range(phi.shape[0]):
        if phi[i] < threshold:
            oc[i] = 0
        else:
            oc[i] =1
    return oc



	
def regress(X, theta):
	# WRITEME: write your code here to complete the routine
    z = np.dot(X,np.transpose(theta))
    return z


def bernoulli_log_likelihood(p, y):
	# WRITEME: write your code here to complete the routine
    return -1.0
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    # WRITEME: write your code here to complete the routine
    m = X.shape[0]
    w=theta[:,1:]
    yhat =sigmoid(regress(X,theta))
    # Here I'm not dividing the bernoulli cross entropy by 2m but just by m. As using 2m leads to gradient also being divided by 2m and results in small values of theta andso if i plot the decision boundary i dont have any thing or a shrinking oval boundary.
    cost = (-np.dot(np.transpose(np.log(yhat)),y))/(m) - (np.dot(np.transpose(np.log(1-yhat)),(1-y)))/(m) + (beta*(np.sum(np.power(w,2))))/(2*m) 
    return cost

def computeGrad(X, y, theta, beta): 
	# WRITEME: write your code here to complete the routine (
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    dL_dfy = None # derivative w.r.t. to model output units (fy)
    yhat = sigmoid(regress(X,theta))
    m = X.shape[0]
    w=theta[:,1:]
    dL_db = (np.dot(np.transpose(yhat-y),X[:,0]))/(m) 
    dL_dw = (np.dot(np.transpose(yhat-y),X[:,1:]))/(m) +  (beta*w)/m
    nabla = np.append(dL_db, dL_dw) # nabla represents the full gradient
    return nabla

def Funcscore(prediction,y):
    accuracy = 0
    n_correct = 0
    for i in range(0,len(predictions)):
        yhat = predictions[i]
        if yhat == y[i]:
            n_correct += 1
    accuracy = n_correct/len(predictions)
    return accuracy


path = os.getcwd() + '/data/prob3.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]
 
x1 = data2['Test 1']  
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.zeros((1,X2.shape[1]))
b = np.array([0])
theta2 = np.append(b, w)
theta2 = theta2.reshape(1,X2.shape[1]+1)


# adding bais unit to feature matrix
bais = np.ones([X2.shape[0],1])
X2_bais = np.append(bais,X2,axis=1)

#Defination loop
L = computeCost(X2_bais, y2, theta2, beta)
print("-1 L = {0}".format(L))
i = 0
cost = [];
cost.append(L)# you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
error = 99999
grad = [];
i = 0

#Training loop 


while ( abs(error) > eps and i < n_epoch  ):  
    G = computeGrad(X2_bais, y2,theta2,beta)
    theta2=theta2 - alpha*G
    # (note: don't forget to override the theta variable...)
    L = computeCost(X2_bais, y2, theta2,beta) # track our loss after performing a single step
    cost.append(L) 
    error = cost[i]- cost[i+1]
    if i % 100 ==0:
        print(" {0} L = {1}".format(i,L))
    i = i +1


# print parameter values found after the search
print("w = ",theta2[:,1:])
print("b = ",theta2[:,0])
print("n_epoch = ",i)
print("Training Error =",cost[-1])

predictions = predict(X2_bais, theta2)
# compute error (100 - accuracy)
err = 0.0

# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
acc=Funcscore(predictions,y2)
err = 100-acc*100

print( 'Error = ',(err))
print('Prediction Accuracy =',(acc*100))

# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
for i in range(1, degree+1):  
    for j in range(0, i+1):
        feat = np.power(xx1, i-j) * np.power(yy1, j)
        if (len(grid_nl) > 0):
            grid_nl = np.c_[grid_nl, feat]
        else:
            grid_nl = feat
bais = np.ones([grid_nl.shape[0],1])
grid_nl_bais = np.append(bais,grid_nl,axis=1)
probs = (regress(grid_nl_bais, theta2)).reshape(xx.shape)


f, ax = plt.subplots(figsize=(8,6))

ax.contour(xx, yy, probs,levels =[0], cmap="RdBu", vmin=0, vmax=.6)
plt.title(r'%sth degree Polynomial Regression Model with '
          '\n'
          r'$\alpha = %s$ and $ \beta =%s$ and threshold = %s'%(degree,alpha,beta,threshold),fontsize =15)
plt.xlabel("X",fontsize=15)
plt.ylabel("Y",fontsize=15)
ax.scatter(x1, x2, c=y2, s=100,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")
plt.tight_layout()

# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.savefig('prob3\Fitted_Model_degree_%s_beta_%s_alpha_%s_threshold_%s.png'%(degree,beta,alpha,threshold))

plt.show()
#Please note this is a script written in python 3 but should work in python 2.7
