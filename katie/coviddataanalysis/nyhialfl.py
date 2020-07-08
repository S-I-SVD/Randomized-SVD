#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 19:03:01 2020

@author: katie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:24:08 2020

@author: katie
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

coviddata = pd.read_csv('/Users/katie/Downloads/nyhialfl_covid.csv')

coviddata = coviddata.drop(columns=['stateFIPS','countyFIPS'])

coviddata_meta = coviddata.loc[:, ['County Name','State']]
 
coviddata = coviddata.drop(columns=['County Name','State'])

coviddata_scaled = (coviddata-coviddata.mean())

u, s, v = np.linalg.svd(coviddata_scaled, full_matrices=True)

var_explained = np.round(s**2/np.sum(s**2), decimals=3)

var_explained = var_explained[:20]

sns.barplot(x=list(range(1,len(var_explained)+1)), y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)
plt.show()

labels= ['SV'+str(i) for i in range(1,5)]
svd_df = pd.DataFrame(u[:,0:4], index=coviddata_meta["County Name"].tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'County Name'}, inplace=True)

labels= ['SV'+str(i) for i in range(1,5)]
svd_df = pd.DataFrame(u[:,0:4], index=coviddata_meta["State"].tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'State'}, inplace=True)

color_dict = dict({'FL':'Black',
                   'AL': 'Red',
                   'HI':'Green',
                   'NY':'Yellow'})
# Scatter plot: SV1 and SV2
sns.scatterplot(x="SV1", y="SV2", hue="State", 
                palette=color_dict, 
                data=svd_df, s=100,
                alpha=0.9)
plt.xlabel('Singular Value 1: {0}%'.format(var_explained[0]*100), fontsize=16)
plt.ylabel('Singular Value 2: {0}%'.format(var_explained[1]*100), fontsize=16)
plt.show()