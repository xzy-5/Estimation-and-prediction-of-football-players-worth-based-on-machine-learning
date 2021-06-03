import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import random
import math

train = pd.read_csv(r'C:\Users\xzy\.spyder-py3\train 处理.csv',encoding = 'utf8')
train = train.drop(['Unnamed: 0'],axis = 1)

plt.subplots(figsize = (100,100))
sns.heatmap(train.corr(),annot = True,xticklabels = train.columns.values.tolist(),yticklabels = train.columns.values.tolist(),cmap = "RdBu",center = 0.0)

#门将数据集
gk_train = (train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)).dropna(axis = 0)
gk_train.isnull().sum()
gk_train.shape

#非门将数据集
ungk_train = (train.drop(['gk'],axis = 1)).dropna(axis = 0)
ungk_train.isnull().sum() 
ungk_train.shape


# 直方图
train.hist(figsize = (25,25))
plt.show()
#验证sliding_tackle
gk_train = (train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)).dropna(axis = 0)
ungk_train = (train.drop(['gk'],axis = 1)).dropna(axis = 0)
gk_train['sliding_tackle'].hist(figsize = (5,5))
plt.title('gk sliding_tackle')
plt.show()
print(np.mean(gk_train['sliding_tackle']))
print(np.median(gk_train['sliding_tackle']))

ungk_train['sliding_tackle'].hist(figsize = (5,5))
plt.title('ungk sliding_tackle')
plt.show()
print(np.mean(ungk_train['sliding_tackle']))
print(np.median(ungk_train['sliding_tackle']))

#验证gk_handing
gk_train['gk_handling'].hist(figsize = (5,5))
plt.title('gk_handling')
plt.show()
print(np.mean(gk_train['gk_handling']))
print(np.median(gk_train['gk_handling']))

ungk_train['gk_handling'].hist(figsize = (5,5))
plt.title('gk_handling')
plt.show()
print(np.mean(ungk_train['gk_handling']))
print(np.median(ungk_train['gk_handling']))





#散点图
for j in range(len(train.columns)):
    sns.lmplot(x=train.columns[j],y='y',data = train,scatter_kws= {'alpha':1/3},fit_reg = False)
    sns.lmplot(x=train.columns[j],y='y',data = train,scatter = False,lowess = True)
    plt.show()
    pass

     #门将
for j in range(len(gk_train.columns)):
    sns.lmplot(x=gk_train.columns[j],y='y',data = gk_train,scatter_kws= {'alpha':1/3},fit_reg = False)
    plt.show()
    pass


     #非门将
for j in range(len(ungk_train.columns)):
    sns.lmplot(x=ungk_train.columns[j],y='y',data = ungk_train,scatter_kws= {'alpha':1/3},fit_reg = False)
    plt.show()
    pass


#热力图
draw1 = train.drop(['club','league','nationality','international_reputation','skill_moves','weak_foot','work_rate_att','work_rate_def','preferred_foot'],axis = 1)
plt.subplots(figsize = (100,100))
sns.heatmap(draw1.corr(),annot = True,xticklabels = draw1.columns.values.tolist(),yticklabels = draw1.columns.values.tolist(),cmap = "RdBu",center = 0.0)
     #门将
draw2 = gk_train.drop(['club','league','nationality','international_reputation','skill_moves','weak_foot','work_rate_att','work_rate_def','preferred_foot'],axis = 1)
plt.subplots(figsize = (100,100))
sns.heatmap(draw2.corr(),annot = True,xticklabels = draw2.columns.values.tolist(),yticklabels = draw2.columns.values.tolist(),cmap = "RdBu",center = 0.0)

data_gk = pd.DataFrame(gk_train.corr())
data_gk.to_excel(r'D:\xzy\数据\gk.xls')
    #非门将
draw3 = ungk_train.drop(['club','league','nationality','international_reputation','skill_moves','weak_foot','work_rate_att','work_rate_def','preferred_foot'],axis = 1)
plt.subplots(figsize = (100,100))
sns.heatmap(draw3.corr(),annot = True,xticklabels = draw3.columns.values.tolist(),yticklabels = draw3.columns.values.tolist(),cmap = "RdBu",center = 0.0)

data2 = pd.DataFrame(ungk_train.corr())
data2.to_excel(r'D:\xzy\数据\ungk.xls')

#小提琴图（离散变量）
list1 = ['international_reputation','skill_moves','weak_foot','work_rate_att','work_rate_def','preferred_foot']
for k in range(len(list1)):
    sns.violinplot(x = list1[k], y= 'y', data = train,split = True)
    plt.show()
    pass

list2 = ['club','league','nationality']
for i in range(len(list2)):
    sns.violinplot(x = list2[i], y= 'y', data = train,split = True)
    plt.show()
    pass




a = train[['club','y']].sample(frac = 0.001,axis = 0)
sns.violinplot(x = a['club'], y = a['y'])
plt.show()

b = train[['league','y']].sample(frac = 0.001,axis = 0)
sns.violinplot(x = b['league'], y = b['y'])
plt.show()

c = train[['nationality','y']].sample(frac = 0.001,axis = 0)
sns.violinplot(x = c['nationality'], y = c['y'])
plt.show()


train.groupby(['club'])

#散点
ungk_train['finishing'] = pd.cut(ungk_train['finishing'].sort_values(),bins =4,labels = ['0','1','2','3'])
sns.lmplot(x='def',y='y',data = ungk_train,hue = 'finishing',scatter_kws= {'alpha':1/3},fit_reg = False)
sns.lmplot(x='def',y='y',data = ungk_train,col = 'finishing',col_wrap = 4,scatter_kws= {'alpha':1/3},fit_reg = False)
