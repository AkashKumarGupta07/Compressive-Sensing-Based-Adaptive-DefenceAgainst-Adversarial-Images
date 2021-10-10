# -*- coding: utf-8 -*-
"""MNIST.ipynb

"""

from numpy import linalg as LA
import numpy as np
import random,math 
from scipy import fftpack
from numpy.random import choice
from scipy.fftpack import dct, idct
import cvxpy as cp
from keras.utils import np_utils
import os.path
import matplotlib.pyplot as plt
from keras.datasets import mnist

from google.colab import drive
drive.mount('/content/drive')



# Input following parameter values 
N          # Dimension of signal for MNIST
k          # Sparsity Assumed for MNIST
t_L0       # No. of pixels perturbed in L0 attack
alpha     # Norm parameters bound
beta      # Norm parameters bound
m         # Norm parameters bound
theta     # Mahalanobis distance parameter
lamda     # weight parameter
Delta    # Stopping Criterion for probability distribution
delt      # Stopping Criterion for l2 norm residual error
gamma     # Tuning Parameter for EXP3 algo
sigma    # Tuning Parameter for EXP3 algo 
I = np.identity(784)
F = dct(I, norm='ortho')
T          #Total number of iterations assumed
initial_iter  # No. of initial iteration assumed to solve optimisation problem 
  
# perturbation level assumed for each action1, action2, action3
eta1
eta2
eta3


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

#Normalizing  data to pixel range [0,1]
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Function to get 2D Inverse Cosine Transform of Image
def get_2d_idct(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

# Loading Clean compressed Image to form the covariance matrix for calculation of Mahalanobis distance

f_np = 'path where clean image residual error is kept'
res_mat=np.load(f_np)
res_mat=res_mat.T
mean_resid=np.mean(res_mat,axis=1)
resid_cov=np.cov(res_mat)
resid_cov_inv=np.linalg.inv(resid_cov)

# Function which returns the Mahalanobis distance of test image taking residual error as parameter

def calc_mahalanobis(res):
    mean_diff = np.asmatrix(np.asarray(res) - np.asarray(mean_resid))
    prod=float(np.sqrt(mean_diff * resid_cov_inv * mean_diff.T))
    return prod

# Action 1  Compressive sampling matching pursuit
# Function takes test image vector and no. of iteration as parameter 
def cosamp(y,iteration):
      
    num_precision = 1e-12
    prev_resid=y
    prev_coeff=np.zeros(N)
    while (iteration > 0):
          
        z=dct(prev_resid.flatten(),norm='ortho')                                                         # Forming signal proxy
        omega = [i for (i, val) in enumerate(z) if val > np.sort(z)[::-1][2*k] and val > num_precision]  # Identifying index of large components
        T = np.union1d(omega, prev_coeff.nonzero()[0])                                                   # Merging both index set
        b = np.dot( np.linalg.pinv(F[:,T]), y )                                                          # Solving Least Square
        best = (abs(b) > np.sort(abs(b))[::-1][k]) & (abs(b) > num_precision)                           # Choosing Top k values only
        T = T[best]
        prev_coeff[T] = b[best]
        prev_resid = y - np.dot(F[:,T], b[best])
        iteration-=1
     
    return prev_coeff,prev_resid

# Action 2 Modofied Basis Pursuit for L_0 Attack
# Function takes test image vector, perturbation level assumed and no. of iteration as parameter 

def action2(y,eta,it2):

    # Defining objective function and other constraints 
    x = cp.Variable(N)
    c = np.zeros((N))
    constraints = [cp.SOC(c.T*x + t_L0*eta, F*x - y)]

    # Form objective.
    obj = cp.Minimize(cp.norm(x,1))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    inc_itr=1
    flag=True
    while(flag==True):
         prob.solve(max_iters=it2+inc_itr,solver='SCS')
         result=x.value
         inc_itr+=5
         try:
            result.shape
            break
         except:
            flag=True
    result[np.argsort(result)[:-k]] = 0 # Choosing top k values only
    return (result)

# Action 3 Standard Basis Pursuit for L_2 Attack
# Function takes test image vector, perturbation level assumed and no. of iteration as parameter 

def action3(y,eta ,it3):
  
    # Defining objective function and other constraints
    x = cp.Variable(N)
    c = np.zeros((N))
    constraints = [cp.SOC(c.T*x + eta, F*x - y)]

    # Form objective.
    obj = cp.Minimize(cp.norm(x,1))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    inc_itr=1
    flag=True
    while(flag==True):
         prob.solve(max_iters=it3+inc_itr,solver='SCS')
         result=x.value
         inc_itr+=5
         try:
            result.shape
            break
         except:
            flag=True
    result[np.argsort(result)[:-k]] = 0 # Choosing top k values only
    return (result)

# Action 4 Standard Basis Pursuit for L_infinity Attack
# Function takes test image vector, perturbation level assumed and no. of iteration as parameter 

def action4(y,eta,it4):

      # Defining objective function and other constraints
      x = cp.Variable(N)
      c = np.zeros((N))
      constraints = [cp.SOC(c.T*x + math.sqrt(N)*eta, F*x - y)]

      # Form objective.
      obj = cp.Minimize(cp.norm(x,1))

      # Form and solve problem.
      prob = cp.Problem(obj, constraints)
      inc_itr=1
      flag=True
      while(flag==True):
         prob.solve(max_iters=it4+inc_itr,solver='SCS')
         result=x.value
         inc_itr+=5
         try:
            result.shape
            break
         except:
            flag=True
      result[np.argsort(result)[:-k]] = 0 # Choosing top k values only
      return (result)

# Compressive Sensing Based Adaptive Defense Algorithm


# Function for getting probability distribution
def get_dist(score):
    score_sum = np.sum(np.exp((sigma*score)))
    return tuple((1.0 - gamma) * (math.exp(sigma*w) / score_sum) + (gamma / len(score)) for w in score)


# Function for choosing action randomly based on probability distribution
def choose(probability_distribution):
    action = choice(4, size=1, p=probability_distribution, replace=False)
    return action


# Function for calculating Residual error
def calc_residual(y,curr_coeff):
    resid=np.subtract(y,get_2d_idct(curr_coeff))
    return resid


# Function for getting feedback signal
def get_feedback(resid, choice):
    if((np.linalg.norm(resid))< alpha):
        # Feedback condition for action 1
        if (choice==0  and np.max(abs(resid)) <m and len((np.where(resid>0.1))[0])<170 and (calc_mahalanobis(np.abs(resid))<theta)):
             return 1
    # Additional Condition for MNIST
    if ( len((np.where(resid>0.1))[0])>250 and (choice==2 or choice ==3 or choice==1)):
            return 1
    else:
        # Checking L0 norm of residual error i.e non-zero elements but since it cannot be exact zero, that's why take atleast greater than 0.5
        if ((np.linalg.norm(resid)) > alpha and len((np.where(abs(resid)>0.5))[0]) < t_L0 and  choice==1):
            return 1
        # Feedback condition for action 3
        if ( (np.linalg.norm(resid)) > alpha and np.max(abs(resid)) < beta  and np.max(abs(resid)) > m and choice==2):
            return 1
        # Feedback condition for action 4
        if ((np.linalg.norm(resid)) >alpha and np.max(abs(resid)) > beta and choice==3):
            return 1
    return 0
            

# Defining EXP3 Algorithm
def exp3(y):
      score = np.zeros(4)
      curr_coeff1=np.zeros(N)
      curr_coeff2=np.zeros(N)
      curr_coeff3=np.zeros(N)
      curr_coeff4=np.zeros(N)
      count_action= np.zeros(4)  # To determine which action is chosen how many times
      resid=y                   
      cosamp_resid=y
      t = 0
      
      while (t<T):
          #print ("Iteration No.- ",t)
          probabilityDistribution = get_dist(score)
          action_taken = choose(probabilityDistribution)
          count_action[action_taken]+=1
          iter=initial_iter+count_action[action_taken]
                  
          if (action_taken==0):
              iter=count_action[action_taken]
              curr_coeff1,cosa_res=cosamp(y,iter)
              cosamp_resid=cosa_res
              
          if (action_taken==1):
              curr_coeff2=action2(y,eta1,iter)
              resid=calc_residual(y,curr_coeff2)
              
          if (action_taken==2):
              curr_coeff3=action3(y,eta2,iter)
              resid=calc_residual(y,curr_coeff3)
            
          if (action_taken==3):
              curr_coeff4=action4(y,eta3,iter)
              resid=calc_residual(y,curr_coeff4)
               
          if (action_taken==0):
               resid=cosa_res
          feedback=get_feedback(resid,action_taken)
          if (feedback==1):
              estimatedreward=lamda / np.array(probabilityDistribution)[action_taken.astype(int)]
          else:
              estimatedreward = -1 / ( 1 - np.array(probabilityDistribution)[action_taken.astype(int)])
          score[action_taken] = score[action_taken] + estimatedreward
          
          # Defining Stopping Criterion 
          if (max(probabilityDistribution) > Delta and LA.norm(resid)< delt):
               break   
          t = t + 1
      type_attack=np.argmax(score)
      if (type_attack==0 or np.max(score) < 0) :
        return curr_coeff1,score,probabilityDistribution
      if type_attack==1:
        return curr_coeff2,score,probabilityDistribution
      if type_attack==2:
        return curr_coeff3,score,probabilityDistribution
      else:
        return curr_coeff4,score,probabilityDistribution

#Loading  Adversaial Test Image
f_np='path to Test Images'
f_in = 'path to index of Test Images'

adver_in = np.load(f_np)
print(adver_in.shape)
index_in = np.load(f_in)
adver_in = np.reshape(adver_in,(len(index_in), N))
nan_img = np.argwhere(np.isnan(adver_in))[:,0]
print(nan_img)
index = []
adver = []
# Removing that images on which foolbox failed to give adversarial image
for i in range(len(adver_in)):
  if( i in nan_img):
    a=1
  else:
    index.append(index_in[i])
    adver.append(adver_in[i])
print(np.argwhere(np.isnan(np.array(adver))))
print(len(index))

# Reconstructing the test images 

noa=len(index)
X_newtrain_adver = np.zeros((noa, 28, 28))
for im in range(noa):
      curr_coef,score,probab_dist = np.array(exp3(adver[im].flatten()))
      X_newtrain_adver [im,:,:]=np.dot(F.T,curr_coef).reshape(28,28)
f_np1 = 'path where you want to save reconstructed test images '
np.save(f_np1,X_newtrain_adver)


