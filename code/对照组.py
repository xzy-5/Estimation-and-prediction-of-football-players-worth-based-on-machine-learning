import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor #决策树
from sklearn.ensemble import RandomForestRegressor #随机森林
import xgboost as xgb #xgboost
from xgboost import plot_importance #xgboost变量重要性 
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from imblearn.over_sampling import SMOTE

plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示

# 导入数据
train = pd.read_csv(r'D:\xzy\数据\data_onehot.csv',encoding = 'utf8')
train = train.drop('Unnamed: 0',axis = 1)

# 查看数据比例
train['is_gk'] = train['gk'] > 0
plt.axes(aspect = 'equal')
counts = train['is_gk'].value_counts()
plt.pie(x = counts,labels = pd.Series(counts.index).map({True:"守门员",False:"非守门员"}))
plt.show()

positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_score'] = train[positions].max(axis = 1)
train = train.drop(['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk'],axis = 1)


# 平衡数据
X,y = train.drop('is_gk',axis = 1),train['is_gk']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state =100)

sample_split = SMOTE(random_state = 100 )
sample_split_X,sample_split_y = sample_split.fit_resample(X_train, y_train)

print(y_train.value_counts()/len(y_train))
print(sample_split_y.value_counts()/len(sample_split_y))

sample_split_y = sample_split_X['y']
sample_split_X = sample_split_X.drop('y',axis = 1)

y_test = X_test['y']
X_test = X_test.drop('y',axis = 1)


# 一.决策树模型

#用守门员数据训练决策树模型
reg = DecisionTreeRegressor(random_state=100)
reg.fit(sample_split_X, sample_split_y)


reg_pred = reg.predict(X_test)
reg_score= r2_score(reg_pred,y_test)
print(reg_score)


# reg模型评估
reg_MAE = mean_absolute_error(y_test,reg_pred)
print("reg模型评价 MAE：\n",reg_MAE)



# (X_gk_train.isnull().sum()).to_csv(r'D:\xzy\数据\see.csv')


# 二.随机森林模型

# 用守门员数据训练随机森林
RF = RandomForestRegressor(random_state=100)
RF.fit(sample_split_X,sample_split_y)
RF_pred = RF.predict(X_test)


#模型预测
RF_MAE = mean_absolute_error(y_test, RF_pred)
print('RF模型评价 MAE:\n',RF_MAE)
RF_score= r2_score(RF_pred,y_test)
print(RF_score)


# 三.xgboost 

# 用守门员数据训练提升树
xgb = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=165,objective='reg:gamma')
xgb.fit(sample_split_X, sample_split_y)
xgb_pred = xgb.predict(X_test)


# 模型预测
xgb_MAE= mean_absolute_error(y_test,xgb_pred)
print('xgb模型评价 MAE:\n',xgb_MAE)
xgb_score= r2_score(xgb_pred,y_test)
print(xgb_score)

# # 变量重要性
# plot_importance(xgb_gk)
# plt.show()

# plot_importance(xgb_ungk)
# plt.show()