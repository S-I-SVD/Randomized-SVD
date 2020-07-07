#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:00:02 2020

@author: jenzyy
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
covid = pd.read_csv("covid_state.csv")

# clean data
state = covid['State']
covid_num = covid.drop(columns=["Unnamed: 0","State"])

# scale each state
covid_scaled = (covid_num - covid_num.mean())/covid_num.std()
covid_scaled.dropna(inplace=True)
df_covid = covid_scaled.loc[:,'2/6/20':].values # select data after 2/6/20

# SVD
u, s, v = np.linalg.svd(df_covid, full_matrices=True)

# scree plot of singular values
var_explained = np.round(s**2/np.sum(s**2), decimals=3)
sns.barplot(x=list(range(1,len(var_explained)+1)),
            y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)

# data frame containing the first two singular vectors
labels= ['SV'+str(i) for i in range(1,3)]
svd_df = pd.DataFrame(u[:,0:2], index=state.tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'State'}, inplace=True)
svd_df.head()
 
# Scatter plot: SV1 and SV2
sns.scatterplot(x="SV1", y="SV2", hue="State", 
                data=svd_df, s=100,
                alpha=0.7)
plt.xlabel('SV 1: {0}%'.format(var_explained[0]*100), fontsize=16)
plt.ylabel('SV 2: {0}%'.format(var_explained[1]*100), fontsize=16)
