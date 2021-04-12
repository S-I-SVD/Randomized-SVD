import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

coviddata = pd.read_csv('/Users/katie/Downloads/fl_al_hi_ny_ga_ca_tx_coviddata.csv')
print(coviddata.shape)

coviddata = coviddata.drop(columns=['stateFIPS','countyFIPS'])
print(coviddata.head(n=3))

#coviddata_meta = coviddata.loc[:, coviddata.columns('County Name','State')]
#print(coviddata_meta.head())

coviddata_meta = coviddata.loc[:, ['County Name','State']]
print(coviddata_meta.head())
 
coviddata = coviddata.drop(columns=['County Name','State'])
print(coviddata.head(n=3))

coviddata_scaled = (coviddata-coviddata.mean())
print(coviddata_scaled.head(n=3))

u, s, v = np.linalg.svd(coviddata_scaled, full_matrices=True)

print(u.shape)
print(s.shape)
print(v.shape)

var_explained = np.round(s**2/np.sum(s**2), decimals=3)
print(var_explained)
print(len(var_explained))

var_explained = var_explained[:20]
print(var_explained)
print(len(var_explained))

sns.barplot(x=list(range(1,len(var_explained)+1)), y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)
plt.show()

#labels= ['SV'+str(i) for i in range(1,3)]
#svd_df = pd.DataFrame(u[:,0:2], index=coviddata_meta["County Name"].tolist(), columns=labels)
#svd_df=svd_df.reset_index()
#svd_df.rename(columns={'index':'County Name'}, inplace=True)
#print(svd_df.head())

print(coviddata_meta)

labels= ['SV'+str(i) for i in range(1,8)]
svd_df = pd.DataFrame(u[:,0:7], index=coviddata_meta["State"].tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'State'}, inplace=True)
print(svd_df.head())
#

##labels= ['SV'+str(i) for i in range(1,3)]
##svd_df = pd.DataFrame(u[:,0:2], index=coviddata_meta["County Name","State"].tolist(), columns=labels)
##svd_df=svd_df.reset_index()
##svd_df.rename(columns={'index':'County Name'}, inplace=True)
##print(svd_df.head())

# specify colors for each continent
color_dict = dict({'FL':'Black',
                   'AL': 'Red',
                   'HI':'Orange',
                   'NY':'Yellow',
                   'GA':'Green',
                   'CA':'Blue',
                   'TX':'Purple'})
# Scatter plot: SV1 and SV2
sns.scatterplot(x="SV1", y="SV2", hue="State", 
                palette=color_dict, 
                data=svd_df, s=100,
                alpha=0.9)
plt.xlabel('Singular Value 1: {0}%'.format(var_explained[0]*100), fontsize=16)
plt.ylabel('Singular Value 2: {0}%'.format(var_explained[1]*100), fontsize=16)
plt.show()

#sns.scatterplot(x="SV1", y="SV2", z="SV3",hue="State", 
#                palette=color_dict, 
#                data=svd_df, s=100,
#                alpha=0.9)
#plt.xlabel('Singular Value 1: {0}%'.format(var_explained[0]*100), fontsize=16)
#plt.ylabel('Singular Value 2: {0}%'.format(var_explained[1]*100), fontsize=16)
#plt.zlabel('Singular Value 3: {0}%'.format(var_explained[1]*100), fontsize=16)
#plt.show()
##
#sv1 = svd_df.loc[:,['SV1']]
#sv2 = svd_df.loc[:,['SV2']]
#sv3 = svd_df.loc[:,['SV3']]
#state = svd_df.loc[:, ['State']]
colors = {'FL':'Black',
                   'AL': 'Red',
                   'HI':'Orange',
                   'NY':'Yellow',
                   'GA':'Green',
                   'CA':'Blue',
                   'TX':'Purple'}
a = list(colors.keys())
b = list(colors.values())
print(a)
print(b)
a1 = tuple(a)
b1 = tuple(b)
print(a1)
print(b1)


#df = pd.DataFrame(dict(sv1=sv1, sv2=sv2, sv3=sv3,state=state))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(svd_df['SV1'], svd_df['SV2'], svd_df['SV3'],c=svd_df['State'].apply(lambda x: colors[x]))
##ax.scatter(df['x'], y, z, c='r', marker='o',color=color_dict)
#
ax.set_xlabel('SV 1: {0}%'.format(var_explained[0]*100), fontsize=8)
ax.set_ylabel('SV 2: {0}%'.format(var_explained[1]*100), fontsize=8)
ax.set_zlabel('SV 3: {0}%'.format(var_explained[2]*100), fontsize=8)
#plt.legend(c,
#           scatterpoints=1,
#           loc='lower left',
#           ncol=3,
#           fontsize=8)
print(svd_df['State'])
#
plt.show()
