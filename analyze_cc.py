# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import ptitprince as pt
import seaborn as sns
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.transforms as transforms
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

os.chdir('/Users/zreagh/Desktop/CamCan_Data')

df = pd.read_csv("camCAN_data_final.csv")
#ts = pd.read_csv("8vid_PS_traces.csv")

ROIs = ["aHPC","pHPC","ANG","PMC","PHC","mPFC","MTG","EVC","AMY"]
region="dmPFC"

#sns.jointplot(x=df['HPC'], y=df['PMC'], kind="hex", color="k")

# RAINCLOUD PLOT!
#sns.set(style="white",context=("poster"),font_scale=1.5)
#
#dx="Group"; dy=region; ort="v"; pal = "ch:.25"; sigma = .2
#ax=pt.RainCloud(x = dx, y = dy, 
#                data = df, 
#                palette = 'colorblind', 
#                width_viol = .6,
#                width_box = .2,
#                figsize = (10,10), orient = 'v',
#                move = .0)
#ax.set_ylim(-0.12,0.4)
#plt.title("{} Activity at Event Boundaries".format(region))
#plt.ylabel("% Signal Change")
#plt.xlabel("Age")
#

# MAIN REGRESSION PLOT!
#region="aHPC"
#sns.set(style="ticks",context=("poster"),font_scale=1.25)
#fig, ax = plt.subplots()
#fig.set_size_inches(10,10)
#sns.regplot(x="Log_Mem_Imm",y=region,data=df,ax=ax,color="gray",line_kws={'color':'purple'})
#plt.title("{} Boundary Activity & Immediate Story Recall".format(region))
#plt.ylabel("% Signal Change: {}".format(region))
#plt.xlabel("Logical Memory Recall: Immediate")
#sns.despine()
#pcorr = stats.pearsonr(df["Log_Mem_Imm"],df["aHPC"])
#print("Pearson R = ", pcorr)

sns.set(style="ticks",context=("poster"),font_scale=1.25)
fig, ax = plt.subplots()
fig.set_size_inches(10,10)
sns.regplot(x="Age",y=region,data=df,ax=ax,color="gray",line_kws={'color':'purple'})
plt.title("{} Activity Within Event".format(region))
plt.ylabel("% Signal Change".format(region))
plt.xlabel("Age")
ax.set_xlim(10,95)
sns.despine()

sns.set(style="ticks",context=("poster"),font_scale=1.25)
fig, ax = plt.subplots()
fig.set_size_inches(6,10)
#ax.set_ylim(0,0.055)

sns.despine()
plt.title("{} Activity Within Event".format(region))
plt.ylabel("% Signal Change")
plt.xlabel("Age Group")
sns.catplot(x="Group",y=region,kind="bar",palette="colorblind",ci=68,data=df,ax=ax)
sns.despine()
#
##g = sns.catplot(x="Group", y=region, kind="point", data=df)
#sns.swarmplot(x="Group", y=region, color="k", size=3, data=df, ax=ax);
#
#sns.regplot(x="ANG",y="pHPC",data=df)
#
##df.corr(method='pearson')
#for ROI in ROIs:
#    region = ROI
#    
#    sns.set(style="ticks",context=("poster"))
#    
#    fig, ax = plt.subplots()
#    fig.set_size_inches(6,8)
##    ax.set_ylim(-0.005,0.16)
#    
#    sns.pointplot(x="Group", y=region, data=df, linestyles='', scale=2, 
#                  color='k', ci=68,errwidth=5, capsize=0.3, markers='o', ax=ax, height=15, palette="colorblind")
#    #produce transform with 5 points offset in x direction
#    offset = transforms.ScaledTranslation(0/72., 0, ax.figure.dpi_scale_trans)
#    trans = ax.collections[0].get_transform()
#    ax.collections[0].set_transform(trans + offset)
#
#    #sns.swarmplot(x="Group", y=region, size=5, data=df, edgecolor="gray", linewidth=.5, ax=ax, palette="ch:.25")
#    #sns.pointplot(x="Group", y=region, data=df, linestyles='--', scale=.5, 
#                  #color='k', errwidth=0, ci=68, ax=ax)
#    plt.ylabel("% Signal Change")
#    plt.xlabel("Age Group")
#    plt.title("{} Activity at Event Boundaries".format(region));
#    
#    plt.show()
#
#plt.title("{} Activity at Event Boundaries".format(region))
#plt.ylabel("% Signal Change")
#plt.xlabel("Age Group")
### ANG-pHPC PLOT!
sns.set(style="ticks",context=("talk"),font_scale=1.25)
sns.lmplot(x="pHPC",y="ANG",hue="Group",palette="colorblind",data=df,height=12,aspect=1)

pcorr = stats.pearsonr(df["Age"],df["dmPFC"])
print("Pearson R = ", pcorr)

spcorr = stats.spearmanr(df["Log_Mem_Del"],df["pHPC"])
print(spcorr)

mod = ols('TP ~ Group', data=df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
print(esq_sm)
print(pairwise_tukeyhsd(df[region],df["Group"]))

mod = ols('TP ~ Sex', data=df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
print(esq_sm)
print(pairwise_tukeyhsd(df[region],df["Sex"]))

### TIME_SERIES PLOT!
#sns.set(style="ticks",context=("poster"),font_scale=1.25)
#fig, ax = plt.subplots()
#fig.set_size_inches(18,12)
#ax.set_ylim(-0.0565,0.181)
#ax = sns.lineplot(x="TR", y="PS", hue="Region", palette="colorblind", err_style="band", data=ts)
#sns.despine()
#ax.legend_.remove()


############## STEPWISE REGRESSION

#X = df[['pHPC','aHPC','PMC','ANG','PHC','MTG','mPFC','EVC','PRC','TP','Motion']].copy()
#y = pd.DataFrame(df['Log_Mem_Imm'])
#
#
#def stepwise_selection(X, y, 
#                       initial_list=[], 
#                       threshold_in=0.04999, 
#                       threshold_out = 0.05, 
#                       verbose=True):
#    """ Perform a forward-backward feature selection 
#    based on p-value from statsmodels.api.OLS
#    Arguments:
#        X - pandas.DataFrame with candidate features
#        y - list-like with the target
#        initial_list - list of features to start with (column names of X)
#        threshold_in - include a feature if its p-value < threshold_in
#        threshold_out - exclude a feature if its p-value > threshold_out
#        verbose - whether to print the sequence of inclusions and exclusions
#    Returns: list of selected features 
#    Always set threshold_in < threshold_out to avoid infinite looping.
#    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
#    """
#    included = list(initial_list)
#    while True:
#        changed=False
#        # forward step
#        excluded = list(set(X.columns)-set(included))
#        new_pval = pd.Series(index=excluded)
#        for new_column in excluded:
#            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
#            new_pval[new_column] = model.pvalues[new_column]
#        best_pval = new_pval.min()
#        if best_pval < threshold_in:
#            best_feature = new_pval.argmin()
#            included.append(best_feature)
#            changed=True
#            if verbose:
#                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
#
#        # backward step
#        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
#        # use all coefs except intercept
#        pvalues = model.pvalues.iloc[1:]
#        worst_pval = pvalues.max() # null if pvalues is empty
#        if worst_pval > threshold_out:
#            changed=True
#            worst_feature = pvalues.argmax()
#            included.remove(worst_feature)
#            if verbose:
#                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
#        if not changed:
#            break
#    return included
#
#result = stepwise_selection(X, y)
#
#print('resulting features:')
#print(result)
#
############### MULTIPLE REGRESSION

from sklearn import linear_model

X = df[['aHPC','PMC','ANG','PHC','MTG','mPFC','EVC','PRC','TP','AMY','Motion','Fluency','Visuospatial','Word_Mem','Log_Mem_Imm','Log_Mem_Del','Age']].copy()
y = pd.DataFrame(df['pHPC'])

regr = linear_model.LinearRegression()
regr.fit(X,y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

X = sm.add_constant(X) # adding a constant

model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)



fig, ax = plt.subplots()
fig.set_size_inches(12,8)

boundary_df = pd.read_csv("boundary_df_new.csv")
sns.distplot(boundary_df['Younger'],  kde=False, label='Younger', bins=60, color='blue')
sns.distplot(boundary_df['Older'],  kde=False, label='Older', bins=60, color='red')

# Plot formatting
plt.legend(prop={'size': 16})
plt.title('Event Segmentation Agreement')
plt.xlabel('Boundary Time')
plt.ylabel('Proportion Responders')