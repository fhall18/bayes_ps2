#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 23:24:31 2021

@author: fhall
"""

#################### IMPORTS HERE ######################################

import arviz as az
from IPython.core.pylabtools import figsize
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from random import sample
import seaborn as sns
import scipy.stats as ss
from scipy import optimize
from statsmodels.api import OLS # has better summary stats than sklearn's OLS

##############################################################################
## Q.3 #######################################################################
##############################################################################

#penguin = pd.read_csv("data/penguins.csv")
penguin = pd.read_csv("penguins.csv")

# 3.1 normalize

min = np.min(penguin['flipper_length_mm'])
max = np.max(penguin['flipper_length_mm'])

flipper_norm = [(val - min) / (max - min) for val in penguin['flipper_length_mm']]
norm = np.array(flipper_norm) - np.array(flipper_norm).mean()
normalized = norm/norm.std()

penguin['flipper_norm'] = normalized
penguin['body_mass_kg'] = penguin['body_mass_g']/1000


print('Normalized flipper outcome: \n','\t std:', '\t',normalized.std(), '\n \t mean: ', '\t',round(normalized.mean(),10)) # std.=1, mean=0

# check to make sure distributions look the same
plt.figure(figsize=(12.5,10))
ax = plt.subplot(211)
plt.hist(penguin.flipper_norm.values, label="normalized flippers",color="#467821")
plt.legend(loc="upper right");
ax = plt.subplot(212)
plt.hist(penguin.flipper_length_mm.values,label="non-normalized flippers")
plt.legend(loc="upper right");

# curious about weight distribution
plt.figure(figsize=(12.5,10))
# plt.hist(penguin.body_mass_kg.values,label="non-normalized",bins = 50)
sns.distplot(penguin.body_mass_kg.values, label="weight density")
plt.legend(loc="upper right");



##############################################################################
# 3.2 FORMULATE PRIORS
##############################################################################

def generate_parameters_prior():
    return [ss.norm(50,np.sqrt(200)).rvs(), # generates alpha
            ss.norm(0,np.sqrt(200)).rvs(),  # generates beta
            ss.uniform(0,100).rvs()]  # generates var


##############################################################################
# 3.3 PRIOR PREDICTIVE
##############################################################################
    
# Create a function that samples X that can be used for the final part
def sample_from_x(num_samples):
    return sample(list(penguin['flipper_norm']),num_samples)


def generate_data(alpha, beta, var, num_samples=10):
    generated_data = []
    X = sample_from_x(num_samples)
    for i in range (num_samples):
        generated_data.append(ss.norm(alpha + beta*X[i], np.sqrt(var)).rvs())
    return generated_data


# PRIOR PREDICTIVE REGRESSION LINES
for i in range(1000):
  params = generate_parameters_prior()
  alpha = params[0]
  beta = params[1]
  var = params[2]

  Y0 = generate_data(alpha, beta, var)
  Y1 = generate_data(alpha, beta, var)

  plt.plot([0, 7], [Y0, Y1], c='gray', alpha=0.1)

plt.xlabel('flipper length')
plt.ylabel('penguin weight')


##############################################################################
# 3.4 POSTERIOR DISTRIBUTION
##############################################################################


def log_posterior(alpha, beta, var, Y, X):
  N = len(X)
  if var > 100 or var < 0:
      return -100000000000000000000
  return (-N * np.log(var) / 2 - np.sum((Y - alpha - beta * X) ** 2) / var +     # log_likelihood
          ss.norm(50, np.sqrt(200)).logpdf(alpha) +                           # log of the prior on alpha
          ss.norm(0, np.sqrt(100)).logpdf(beta))                              # log of the prior on beta
            

def minus_log_posterior(theta):
    alpha = theta[0]
    beta = theta[1]
    var = theta[2]
    Y = penguin['body_mass_kg']
    X = penguin['flipper_norm']
    return - log_posterior(alpha, beta,var,Y,X)


fit = optimize.minimize(minus_log_posterior, [50, 0, 10])


# fit approx.
MAP = fit['x']
hess_inv  = fit['hess_inv']
approx = ss.multivariate_normal(MAP, hess_inv)

##############################################################################
# 3.5 SUMMARIZING WITH SAMPLES
##############################################################################

samples = approx.rvs(1000)

plt.figure(figsize=(12.5,10))
plt.hist(samples[:,1])
plt.ylabel('Distribution')
plt.xlabel(r'$\beta$')
sns.despine()

plt.figure(figsize=(12.5,10))
sns.pairplot(pd.DataFrame(samples, columns=['alpha', 'beta', 'var']))


##############################################################################
# 3.6 POSTERIOR PREDICTIVE SIMULATIONS
##############################################################################


# data points and regression lines
plt.figure(figsize=(12.5,10))
for i in range(10):  # 10 data sets
  params = approx.rvs()
  alpha = params[0]
  beta = params[1]
  var = params[2]

  Y0 = generate_data(alpha, beta, var, 50)  # simulate 50 measurements at 0
  Y1 = generate_data(alpha, beta, var, 50)  # simulate 50 measurements at 1

  plt.plot([alpha, alpha + beta], c='blue', alpha=0.1, zorder=-100)
  plt.plot([alpha, alpha + beta], c='blue', alpha=0.1, zorder=-100)
   # plt.scatter([0] * 50 + [1] * 50, np.concatenate((Y0, Y1)), c='gray', alpha=0.1, zorder=-100)

plt.xlabel('Flipper Size')
plt.ylabel('Weight')
plt.scatter(penguin['flipper_norm'], penguin['body_mass_kg'], s=2, c='red')



##############################################################################
# 3 ARCHIVE
##############################################################################




# # lm = linear_model.LinearRegression()
# x = penguin.flipper_norm.values.reshape(len(penguin),1)
# y = penguin.body_mass_kg.values.reshape(len(penguin),1)
# intercept = np.full((len(penguin),1),1)

# df = pd.DataFrame(intercept)
# df['x'] = x
# df.rename(columns={0: 'intercept'})

# lm = OLS(y, df)
# results = lm.fit()
# results.summary()


# resolution = 100
# alpha_grid = np.linspace(4.160, 4.244, resolution) # estimate where mu should fall
# beta_grid = np.linspace(0.656, 0.740, resolution) # estimate where var should fall
# var_grid = np.linspace(25, 35, resolution) # estimate where var should fall
