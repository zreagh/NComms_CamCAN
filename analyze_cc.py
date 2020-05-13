# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import seaborn as sns
import ptitprince as pt
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data_dir = '/Users/zreagh/Desktop/CamCAN_Analyses'
fig_dir = data_dir+'/figs/'
os.chdir(data_dir)

df = pd.read_csv("camCAN_data_final.csv")
#ts = pd.read_csv("8vid_PS_traces.csv")

ROIs = ["aHPC","pHPC","ANG","PMC","PHC","mPFC","MTG","VC","AMY","PRC","TP"]
region="pHPC"

# Loop
for ROI in ROIs:
    # set our ROI
    region = ROI
    # create the scatter plot
    sns.set(style="ticks",context=("poster"),font_scale=1.25)
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    sns.regplot(x="Age",y=region,data=df,ax=ax,color="gray",line_kws={'color':'purple'})
    ax.set_xlim(15,95)
    sns.despine()
    plt.title("{} Activity at Event Boundaries".format(region),y=1.02)
    plt.ylabel("% Signal Change\n(Boundary - Within-Event)".format(region))
    plt.xlabel("Age")
    plt.savefig(fig_dir+'{}_corr.pdf'.format(region),format='pdf',dpi=300,bbox_inches="tight")
    plt.show()
    # print out the Pearson correlation coefficient and p-value
    pcorr = stats.pearsonr(df["Age"],df["pHPC"])
    print("{} Pearson R = ".format(region), pcorr)
    
    # create raincloud plots
    sns.set(style="ticks",context=("poster"),font_scale=1.25)
    fig, ax = plt.subplots()
    fig.set_size_inches(6,10)
    dx="Group"; dy=region; ort="v"; pal = "colorblind"; sigma = .25
    ax=pt.RainCloud(x = dx, y = dy, 
                    data = df, 
                    palette = 'colorblind', 
                    width_viol = .5,
                    width_box = .25,
                    orient = ort,
                    move = .0,
                    bw = sigma,
                    showfliers = False)
    plt.title("{} Activity at Event Boundaries".format(region),y=1.02)
    plt.ylabel("% Signal Change\n(Boundary - Within-Event)")
    plt.xlabel("Group")
    sns.despine()
    plt.savefig(fig_dir+'{}_raincloud.pdf'.format(region),format='pdf',dpi=300,bbox_inches="tight")
    # run ANOVAs with posthoc Tukey HSD comparisons
    print("\n{} One-Way ANOVA:".format(region))
    mod = ols('{} ~ Group'.format(region), data=df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
    print("{} group effect eta-squared: ".format(region), esq_sm)
    print(pairwise_tukeyhsd(df[region],df["Group"]))


###### BAR PLOTS - DEPRECATED ######
#sns.set(style="ticks",context=("poster"),font_scale=1.25)
#fig, ax = plt.subplots()
#fig.set_size_inches(6,10)
#ax.set_ylim(0,0.055)

#sns.despine()
#plt.title("{} Activity Within Event".format(region))
#plt.ylabel("% Signal Change")
#plt.xlabel("Age Group")
#sns.catplot(x="Group",y=region,kind="bar",palette="colorblind",ci=68,data=df,ax=ax)
#sns.despine()
######


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

# correlations between hippocampal activity and neuropsych tests

for region in ['aHPC', 'pHPC']:
    # logical memory: immediate
    sns.set(style="ticks",context=("poster"),font_scale=1.25)
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    sns.regplot(x="Log_Mem_Imm",y=region,data=df,ax=ax,color="gray",line_kws={'color':'purple'})
    plt.title("{} Boundary Activity & Immediate Story Recall".format(region),y=1.02)
    plt.ylabel("% Signal Change\n(Boundary - Within-Event)")
    plt.xlabel("Logical Memory Recall: Immediate")
    sns.despine()
    plt.savefig(fig_dir+'{}_LogMemImm.pdf'.format(region),format='pdf',dpi=300,bbox_inches="tight")
    pcorr = stats.pearsonr(df["Log_Mem_Imm"],df[region])
    print("\n{} & Logical Memory Immediate, Pearson R = ".format(region), pcorr)
    # logical memory: delayed
    sns.set(style="ticks",context=("poster"),font_scale=1.25)
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    sns.regplot(x="Log_Mem_Del",y=region,data=df,ax=ax,color="gray",line_kws={'color':'purple'})
    plt.title("{} Boundary Activity & Delayed Story Recall".format(region),y=1.02)
    plt.ylabel("% Signal Change\n(Boundary - Within-Event)")
    plt.xlabel("Logical Memory Recall: Immediate")
    sns.despine()
    plt.savefig(fig_dir+'{}_LogMemDel.pdf'.format(region),format='pdf',dpi=300,bbox_inches="tight")
    pcorr = stats.pearsonr(df["Log_Mem_Del"],df[region])
    print("{} & Logical Memory Delayed, Pearson R = ".format(region), pcorr)


## Event boundaries plot, not in main text
#fig, ax = plt.subplots()
#fig.set_size_inches(12,8)
#
#boundary_df = pd.read_csv("boundary_df_new.csv")
#sns.distplot(boundary_df['Younger'],  kde=False, label='Younger', bins=60, color='blue')
#sns.distplot(boundary_df['Older'],  kde=False, label='Older', bins=60, color='red')
#
## Plot formatting
#plt.legend(prop={'size': 16})
#plt.title('Event Segmentation Agreement')
#plt.xlabel('Boundary Time')
#plt.ylabel('Proportion Responders')