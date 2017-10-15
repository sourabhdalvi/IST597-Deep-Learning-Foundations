import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
#Please note this is a script written in python 3 but should work in python 2.7
'''
IST 597: Foundations of Deep Learning
Problem 2: Polynomial Regression & 

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

# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree = 15 # p, order of model
beta = 1# regularization coefficient
alpha = 1 # step size coefficient
eps = 0.000001 # controls convergence criterion
n_epoch = 10000 # number of epochs (full passes through the dataset)
#Please note this is a script written in python 3 but should work in python 2.7
# begin simulation

def regress(X, theta):
	# WRITEME: write your code here to complete the routine
    yhat = np.dot(X,np.transpose(theta));
    return yhat

def gaussian_log_likelihood(mu, y):
	# WRITEME: write your code here to complete the sub-routine
    gll =0;
    return gll
	
def computeCost(X, y, theta,beta): # loss is now Bernoulli cross-entropy/log likelihood
	# WRITEME: write your code here to complete the routine
    m = X.shape[0];
    yhat = regress(X,theta);
    cost =sum(pow(yhat-y,2))/(2*m) + (beta*sum(sum(pow(w,2))))/(2*m);
    return cost

def computeGrad(X, y, theta,beta): 
	# WRITEME: write your code here to complete the routine
	# NOTE: you do not have to use the partial derivative symbols below, they are there to guide your thinking)
    yhat = regress(X,theta);
    w=theta[:,1:]
    m = X.shape[0]
    dL_db = (np.dot(np.transpose(yhat-y),X[:,0]))/m 
    dL_dw = (np.dot(np.transpose(yhat-y),X[:,1:]))/m +  (beta*w)/m
    nabla = np.append(dL_db, dL_dw) ;# nabla represents the full gradient
    return nabla


path = os.getcwd() + '/data/prob2.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
def MAP(X,degree):
    for i in range(degree+1):
        if i == 0 :
            ploy=pow(X,0);
        else :
            ploy = np.append(ploy,pow(X,i),axis=1)
    return ploy

X_poly=MAP(X,degree)

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X_poly.shape[1]-1))
b = np.array([[0]])
theta = np.append(b, w)
theta = theta.reshape(1,X_poly.shape[1])

L = computeCost(X_poly, y, theta, beta)
print("-1 L = {0}".format(L))
i = 0
cost = [];
cost.append(L)# you can use this list variable to help you create the loss versus epoch plot at the end (if you want)
error = 99999

while ( abs(error) > eps and i < n_epoch  ): 
    grad = computeGrad(X_poly, y,theta,beta)
    theta=theta -alpha*grad
    # (note: don't forget to override the theta variable...)
    L = computeCost(X_poly, y, theta,beta) # track our loss after performing a single step
    cost.append(L) 
    error = cost[i]- cost[i+1]
    if i % 100==0:
        print(" {0} L = {1}".format(i,L))
    
    i = i +1
   
    
print("w = ",theta[:,1:])
print("b = ",theta[:,0])
print("n_epoch = ",i)
print("Training Error =",cost[-1])

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))


# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you could use a loop and array concatenation)
X_feat_poly = MAP(X_feat,degree)

plt.figure(figsize=(8,6))
plt.plot(X_test, regress(X_feat_poly, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=50, label="Samples")

plt.xlabel("X",fontsize=15)
plt.ylabel("Y",fontsize=15)
plt.title(r'%sth degree Polynomial Regression Model with '
          '\n'
          r'$\alpha = %s$ and $\beta =%s$'%(degree,alpha,beta),fontsize =15)
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")
# WRITEME: write your code here to save plot to disk (look up documentation/inter-webs for matplotlib)
plt.savefig('prob2\Fitted_Model_degree_%s_beta_%s_alpha_%s.png'%(degree,beta,alpha))
plt.show()

# Loss function plot
plt.figure(figsize=(8,6))
plt.plot(cost)

plt.xlabel("Epoch",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.title(r'%sth degree Polynomial Regression Model with'
          '\n'
          r'$\alpha = %s$ and $\beta =%s$'%(degree,alpha,beta),fontsize =15)
plt.savefig('prob2\plot_Loss_degree_%s_beta_%s_alpha_%s.png'%(degree,beta,alpha))
plt.show()
#Please note this is a script written in python 3 but should work in python 2.7
