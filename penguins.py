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
import seaborn as sns
import scipy.stats as ss
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS # has better summary stats than sklearn's OLS

##############################################################################
## Q.3 #######################################################################
##############################################################################

penguin_raw = pd.read_csv("data/penguins.csv")

# 3.1 normalize
flipper_mean = penguin_raw['flipper_length_mm'].mean()
dist_norm = penguin_raw['flipper_length_mm'] - flipper_mean

min = np.min(penguin_raw['flipper_length_mm'])
max = np.max(penguin_raw['flipper_length_mm'])

flipper_norm = [(val - min) / (max - min) for val in penguin_raw['flipper_length_mm']]
# scale = np.frompyfunc(lambda x, min, max: (x - min) / (max - min), 3, 1)
norm = np.array(flipper_norm) - np.array(flipper_norm).mean()
normalized = norm/norm.std()

penguin_raw['flipper_norm'] = normalized
penguin_raw['body_mass_kg'] = penguin_raw['body_mass_g']/1000


print("std:", normalized.std(), 'mean: ', normalized.mean()) # std.=1, mean=0

# check to make sure distributions look the same
plt.figure(figsize=(12.5,10))
ax = plt.subplot(211)
plt.hist(penguin_raw.flipper_norm.values, label="normalized flippers",color="#467821")
plt.legend(loc="upper right");
ax = plt.subplot(212)
plt.hist(penguin_raw.flipper_length_mm.values,label="non-normalized flippers")
plt.legend(loc="upper right");

# curious about weight distribution
plt.figure(figsize=(12.5,10))
# plt.hist(penguin_raw.body_mass_kg.values,label="non-normalized",bins = 50)
sns.distplot(penguin_raw.body_mass_kg.values, label="weight density")
plt.legend(loc="upper right");

##############################################################################
# 3.2 FORMULATE PRIORS
##############################################################################


# lm = linear_model.LinearRegression()
x = penguin_raw.flipper_norm.values.reshape(len(penguin_raw),1)
y = penguin_raw.body_mass_kg.values.reshape(len(penguin_raw),1)
intercept = np.full((len(penguin_raw),1),1)

df = pd.DataFrame(intercept)
df['intercept'] = intercept
df['x'] = x

lm = OLS(y, df)
results = lm.fit()
results.summary()


resolution = 100
alpha_grid = np.linspace(4.160, 4.244, resolution) # estimate where mu should fall
beta_grid = np.linspace(0.656, 0.740, resolution) # estimate where var should fall
var_grid = np.linspace(25, 35, resolution) # estimate where var should fall



# with normal distribution
def create_model_(data, alpha, beta, var):
    # Create linear regression object
    lm = linear_model.LinearRegression()
    x = penguin_raw.flipper_norm.values.reshape(len(data),1)
    y = penguin_raw.body_mass_kg.values.reshape(len(data),1)

    lm.fit(x.reshape(len(x),1), y.reshape(len(y),1))
    print('intercept:', lm.intercept_)
    print('slope:', lm.coef_[0])
    intercept = lm.intercept_[0]
    slope = lm.coef_[0]

    plt.scatter(x, y, color='black') # plot
    plt.plot(x, lm.predict(x), color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    with pm.Model() as model: 
    
        alpha = pm.Normal('alpha',testval=intercept) # check if there is a variance input
        # sufficient statistics
        n = len(data)
        k = np.sum(data)
        # binominal = pm.Binomial('binomial', n, theta, observed=k)
        
    return model,n,k



model = create_model_(penguin_raw)



##############################################################################
# 3.5 SUMMARIZING WITH SAMPLES
##############################################################################
with model:
    p_trace = pm.sample(5000) # draw 10000 posterior samples

    theta_trace = p_trace['alpha'][1000:]
    theta_trace = p_trace['beta'][1000:]


##############################################################################
# 3.3 FORMULATE PRIORS
##############################################################################


