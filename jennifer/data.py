#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:00:02 2020

@author: jenzyy

data source: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import dataset
#covid = pd.read_csv("../data/covid data/covid_state.csv")
covid = pd.read_csv("../data/covid data/covid_state_new.csv")

# clean data
state_name = covid['State']
covid_num = covid.drop(columns=["Unnamed: 0","State"])

# scale each state
covid_scaled = (covid_num - covid_num.mean())/(covid_num.std())
covid_scaled.dropna(inplace=True,axis='columns')
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
svd_df = pd.DataFrame(u[:,0:2], index=state_name.tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'State'}, inplace=True)
svd_df.head()
 
# take in party variable
state = pd.read_csv("../data/covid data/state_party.csv")
svd_df['party'] = state['party']

# Scatter plot: SV1 and SV2
sns.scatterplot(x="SV1", y="SV2", hue="party", 
                data=svd_df, s=100,
                alpha=0.7)
for j in range(svd_df.shape[0]):  
    plt.text(svd_df['SV1'][j],svd_df['SV2'][j],svd_df['State'][j])
plt.xlabel('SV 1: {0}%'.format(var_explained[0]*100), fontsize=16)
plt.ylabel('SV 2: {0}%'.format(var_explained[1]*100), fontsize=16)
plt.text

# data frame containing the first three singular values
labels1= ['SV'+str(i) for i in range(1,4)]
svd_df1 = pd.DataFrame(u[:,0:3], index=state_name.tolist(), columns=labels1)
svd_df1=svd_df1.reset_index()
svd_df1.rename(columns={'index':'State'}, inplace=True)
svd_df1.head()

# take in party variable
svd_df1['party']=(state['party']=="Republican")

# 3D Scatter plot: SV1, SV2, and SV3
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection = '3d')
#ax.scatter(svd_df1['SV1'],svd_df1['SV2'],svd_df1['SV3'],c=svd_df1['party'],cmap = 'coolwarm',)    
for j in range(svd_df1.shape[0]):
    if svd_df1.loc[j,'party']:
        ax.scatter(svd_df1['SV1'][j],svd_df1['SV2'][j],svd_df1['SV3'][j], marker = 'x', color = 'r')
    else:
        ax.scatter(svd_df1['SV1'][j],svd_df1['SV2'][j],svd_df1['SV3'][j], marker = 'o', color = 'b')    
    ax.text(svd_df1['SV1'][j],svd_df1['SV2'][j],svd_df1['SV3'][j],svd_df1['State'][j])
ax.view_init(25,20)
ax.legend
plt.show()
