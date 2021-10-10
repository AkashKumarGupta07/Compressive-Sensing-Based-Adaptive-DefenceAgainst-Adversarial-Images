# -*- coding: utf-8 -*-
"""Cifar_10.ipynb

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
from keras.datasets import cifar10

from google.colab import drive
drive.mount('/content/drive')

# Input following parameter values 

N        # Dimension of vector for Cifar10
k        # Sparsity Assumed for Cifar10
t_L0     # No. of pixels perturbed in L0 attack
alpha    # Norm parameters bound
beta     # Norm parameters bound
m        # Norm parameters bound
theta1    # Mahalanobis distance parameter channel 1
theta2    # Mahalanobis distance parameter channel 2
theta3    # Mahalanobis distance parameter channel 3
lamda    # weight parameter
Delta    # Stopping Criterion for probability distribution
delt     # Stopping Criterion for l2 norm residual error
gamma    # Tuning Parameter for EXP3 algo
sigma    # Tuning Parameter for EXP3 algo 
I = np.identity(N)
F = dct(I, norm='ortho')
T        #Total number of iterations assumed
initial_iter            # No. of initial iteration assumed to solve optimisation problem 
# perturbation level assumed for each action1, action2, action3
eta1
eta2
eta3


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape(-1, 32, 32, 3)
test_images = test_images.reshape(-1, 32, 32, 3)

# Normalizing  data to pixel range [0,1]
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
ch1=res_mat[:,:,0]
ch2=res_mat[:,:,1]
ch3=res_mat[:,:,2]
print(ch1.shape)
mean_resid_ch1=np.mean(ch1,axis=1)
mean_resid_ch2=np.mean(ch2,axis=1)
mean_resid_ch3=np.mean(ch3,axis=1)
print(mean_resid_ch1.shape)
resid_cov_ch1=np.cov(ch1)
resid_cov_ch2=np.cov(ch2)
resid_cov_ch3=np.cov(ch3)
print(resid_cov_ch1.shape)
resid_cov_inv_ch1=np.linalg.norm(resid_cov_ch1)
resid_cov_inv_ch2=np.linalg.norm(resid_cov_ch2)
resid_cov_inv_ch3=np.linalg.norm(resid_cov_ch3)

# Functions for each channel which returns the Mahalanobis distance of test image taking residual error as parameter

def calc_mahalanobis_ch1(res):
    mean_diff = np.asmatrix(np.asarray(res) - np.asarray(mean_resid_ch1))
    prod=float(np.sqrt(mean_diff * resid_cov_inv_ch1 * mean_diff.T))
    return prod
def calc_mahalanobis_ch2(res):
    mean_diff = np.asmatrix(np.asarray(res) - np.asarray(mean_resid_ch2))
    prod=float(np.sqrt(mean_diff * resid_cov_inv_ch2 * mean_diff.T))
    return prod
def calc_mahalanobis_ch3(res):
    mean_diff = np.asmatrix(np.asarray(res) - np.asarray(mean_resid_ch3))
    prod=float(np.sqrt(mean_diff * resid_cov_inv_ch3 * mean_diff.T))
    return prod

# Action 1  Compressive sampling matching pursuit
# Function takes test image vector and no. of iteration as parameter 
def cosamp(y,iteration):
    
    num_precision = 1e-12
    prev_resid=y
    prev_coeff=np.zeros(N)
    while (iteration > 0):  
        z=dct(prev_resid.flatten(),norm='ortho')                                                            # Forming signal proxy
        omega = [i for (i, val) in enumerate(z) if val > np.sort(z)[::-1][2*k] and val > num_precision]     # Identifying index of large components
        T = np.union1d(omega, prev_coeff.nonzero()[0])                                                      # Merging both index set
        b = np.dot( np.linalg.pinv(F[:,T]), y )                                                             # Solving Least Square
        best = (abs(b) > np.sort(abs(b))[::-1][k]) & (abs(b) > num_precision)                              # Choosing Top k values only
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
def calc_residual(y,recon_img):
    resid=np.subtract(y,recon_img)
    return resid


# Function for getting feedback 
def get_feedback(resid, choice):
    res_ch=resid.reshape(32,32,3)
    mh1=calc_mahalanobis_ch1(np.abs(res_ch[:,:,0]).flatten())
    mh2=calc_mahalanobis_ch2(np.abs(res_ch[:,:,1]).flatten())
    mh3=calc_mahalanobis_ch3(np.abs(res_ch[:,:,2]).flatten())  
    
    if((np.linalg.norm(resid))< alpha or  mh1 < theta1 or mh2 < theta2 or mh3 < theta3):
         # Feedback condition for action 1
        if (choice==0  and np.max(abs(resid)) <m ):
             return 1
    # Feedback condition for action 2
    if ((np.linalg.norm(resid)) > alpha  and len((np.where(abs(resid)>0.5))[0]) < t_L0 and  choice==1): 
        return 1
    # Feedback condition for action 3
    if ( (np.linalg.norm(resid)) > alpha and np.max(abs(resid)) < beta   and np.max(abs(resid)) > m and choice==2):
        return 1
    # Feedback condition for action 4
    if ((np.linalg.norm(resid)) >alpha and np.max(abs(resid)) > beta and choice==3):
        return 1
    return 0
            
# Defining EXP3 Algorithm
def exp3(y):
      score = np.zeros(4)
      curr_coeff=np.zeros(N)
      count_action= np.zeros(4)  # To determine which action is chosen how many times
      cosamp_resid=np.zeros((N,3))
      t = 0
      
      Result_Array1=np.zeros((32,32,3))
      Result_Array2=np.zeros((32,32,3))
      Result_Array3=np.zeros((32,32,3))
      Result_Array4=np.zeros((32,32,3))
      while (t<T):
          #print ("Iteration No.- ",t)
          probabilityDistribution = get_dist(score)
          action_taken = choose(probabilityDistribution)
          count_action[action_taken]+=1
          iter=initial_iter+count_action[action_taken]
          if (action_taken==0):
              iter=4+count_action[action_taken]
              for ch in range(3):
                    curr_coeff,re=cosamp(np.array(y[:,:,ch].flatten()),iter)
                    cosamp_resid[:,ch]=abs(re)
                    Result_Array1[:,:,ch]=idct(curr_coeff,norm='ortho').reshape(32,32)
          if (action_taken==1):
              for ch in range(3):
                    curr_coeff=action2(np.array(y[:,:,ch].flatten()),eta1,iter)
                    Result_Array2[:,:,ch]=idct(curr_coeff,norm='ortho').reshape(32,32)
                    resid=calc_residual(y.flatten(),Result_Array2.flatten())
 
          if (action_taken==2):
              for ch in range(3):
                    curr_coeff=action3(np.array(y[:,:,ch].flatten()),eta2,iter)
                    Result_Array3[:,:,ch]=idct(curr_coeff,norm='ortho').reshape(32,32)
                    resid=calc_residual(y.flatten(),Result_Array3.flatten())
          if (action_taken==3):
              for ch in range(3):
                    curr_coeff=action4(np.array(y[:,:,ch].flatten()),eta3,iter)
                    Result_Array4[:,:,ch]=idct(curr_coeff,norm='ortho').reshape(32,32)
                    resid=calc_residual(y.flatten(),Result_Array4.flatten())
          if (action_taken==0):
                resid=cosamp_resid
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
        return Result_Array1,score,probabilityDistribution
      if type_attack==1:
        return Result_Array2,score,probabilityDistribution
      if type_attack==2:
        return Result_Array3,score,probabilityDistribution
      else:
        return Result_Array4,score,probabilityDistribution

#Loading  Test Image

f_np = 'path to adversarial Images'
f_ind='path to index of adversarial Images'
index_in = np.load(f_ind)
adver_in = np.load(f_np).reshape(index_in.shape[0],32,32,3)
print(adver_in.shape)
adver = np.reshape(adver_in, (32, 32, 3, len(index_in)))
nan_img = np.argwhere(np.isnan(adver_in))[:,0]
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
adver=np.array(adver)/255
adver=adver.reshape(adver.shape[0],32,32,3)

# Reconstructing the test images 

noa=len(index)
X_newtest_adver=np.zeros((noa,32,32,3))
for im in range(noa):
      recon_img, score,probab_dist=np.array(exp3(adver[im]))
      X_newtest_adver[im,:,:,:]=recon_img
      
f_np ='path where you want to save reconstructed test images '
np.save(f_np,X_newtest_adver)
