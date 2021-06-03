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

plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示
# 导入数据
train = pd.read_csv(r'D:\xzy\数据\data_onehot.csv',encoding = 'utf8')
train = train.drop('Unnamed: 0',axis = 1)

# 根据缺失值，划分守门员和非守门员
gk_train = (train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)).dropna(axis = 0)

ungk_train = (train.drop(['gk'],axis = 1)).dropna(axis = 0)

# 拆分训练集
X_gk,y_gk = gk_train.drop('y',axis = 1),gk_train['y']
X_gk_train,X_gk_test,y_gk_train,y_gk_test = train_test_split(X_gk,y_gk,test_size = 0.2,random_state =100)

X_ungk,y_ungk = ungk_train.drop('y',axis = 1),ungk_train['y']
X_ungk_train,X_ungk_test,y_ungk_train,y_ungk_test = train_test_split(X_ungk,y_ungk,test_size = 0.2,random_state =100)
 
# 一.决策树模型
# 建立决策树模型

#用守门员数据训练决策树模型
reg_gk = DecisionTreeRegressor(random_state=100)
reg_gk.fit(X_gk_train, y_gk_train)
reg_pred_gk = reg_gk.predict(X_gk_test)

#用非守门员数据训练决策树模型
reg_ungk = DecisionTreeRegressor(random_state=100)
reg_ungk.fit(X_ungk_train, y_ungk_train)
reg_pred_ungk = reg_ungk.predict(X_ungk_test)

# reg模型评估
reg_MAE_gk = mean_absolute_error(y_gk_test,reg_pred_gk)
print("reg模型评价 MAE_gk：\n",reg_MAE_gk)
print('score_gk:\n' ,r2_score(reg_pred_gk,y_gk_test))

reg_MAE_ungk = mean_absolute_error(y_ungk_test,reg_pred_ungk)
print("reg模型评价 MAE_ungk：\n",reg_MAE_ungk)
print('score_ungk:\n',r2_score(reg_pred_ungk,y_ungk_test))


reg_MAE = (((abs(reg_pred_gk - y_gk_test)).sum())+((abs(reg_pred_ungk - y_ungk_test)).sum()))/(len(y_gk_test)+len(y_ungk_test))
print('reg模型评价 MAE:\n',reg_MAE)
# 交叉验证
max_depth = [16,18,20,24,26]
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8,10,12]
paramers = {'min_samples_split':min_samples_split,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf}
grid_dicateg = GridSearchCV(estimator = DecisionTreeRegressor(),param_grid=paramers,cv = 5)
grid_dicateg.fit(X_gk_train,y_gk_train)
grid_dicateg.best_params_
reg_cv5_gk = grid_dicateg.predict(X_gk_train)
reg_MAE_gk_cv5 = mean_absolute_error(y_gk_train,reg_cv5_gk)
print('xgb模型评价 MAE_gk:\n',reg_MAE_gk_cv5)

max_depth = [18,20,22,24,26,28]
min_samples_split = [2,4,6,8]
min_samples_leaf = [2,4,8,10,12]
paramers = {'min_samples_split':min_samples_split,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf}
grid_dicateg = GridSearchCV(estimator = DecisionTreeRegressor(),param_grid=paramers,cv = 5)
grid_dicateg.fit(X_ungk_train,y_ungk_train)
grid_dicateg.best_params_
reg_cv5_ungk = grid_dicateg.predict(X_ungk_train)
reg_MAE_ungk_cv5 = mean_absolute_error(y_ungk_train,reg_cv5_ungk)
print('reg模型评价 MAE_ungk:\n',reg_MAE_ungk_cv5)

reg_MAE = (((abs(y_gk_train - reg_cv5_gk)).sum())+((abs(y_ungk_train - reg_cv5_ungk)).sum()))/(len(y_gk_train)+len(y_ungk_train))
print('reg模型评价 MAE:\n',reg_MAE)

# 二.随机森林模型

# 用守门员数据训练随机森林
RF_gk = RandomForestRegressor(random_state=100)
RF_gk.fit(X_gk_train,y_gk_train)
RF_pred_gk = RF_gk.predict(X_gk_test)

# 用非守门员数据训练随机森林
RF_ungk = RandomForestRegressor(random_state=100)
RF_ungk.fit(X_ungk_train,y_ungk_train)
RF_pred_ungk = RF_ungk.predict(X_ungk_test)

#模型预测
RF_MAE_gk = mean_absolute_error(y_gk_test, RF_pred_gk)
print('RF模型评价 MAE_gk:\n',RF_MAE_gk)
print('score_ungk:\n',r2_score(RF_pred_gk,y_gk_test))

RF_MAE_ungk = mean_absolute_error(y_ungk_test, RF_pred_ungk)
print('RF模型评价 MAE_ungk:\n',RF_MAE_ungk)
print('score_ungk:\n',r2_score(RF_pred_ungk,y_ungk_test))

RF_MAE = (((abs(RF_pred_gk - y_gk_test)).sum())+((abs(RF_pred_ungk - y_ungk_test)).sum()))/(len(y_gk_test)+len(y_ungk_test))
print('RF模型评价 MAE:\n',RF_MAE)


# 三.xgboost 

# 用守门员数据训练提升树
xgb_gk = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=185,objective='reg:linear')
xgb_gk.fit(X_gk_train, y_gk_train)

xgb_pred_gk = xgb_gk.predict(X_gk_test)
xgb_MAE_gk = mean_absolute_error(y_gk_test,xgb_pred_gk)
print('xgb模型评价 MAE_gk:\n',xgb_MAE_gk)
# 用非守门员数据训练提升树
xgb_ungk = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=230,objective='reg:linear')
xgb_ungk.fit(X_ungk_train, y_ungk_train)

xgb_pred_ungk = xgb_ungk.predict(X_ungk_test)
xgb_MAE_ungk = mean_absolute_error(y_ungk_test,xgb_pred_ungk)
print('xgb模型评价 MAE_ungk:\n',xgb_MAE_ungk)

xgb_MAE = (((abs(y_gk_test - xgb_pred_gk)).sum())+((abs(y_ungk_test - xgb_pred_ungk)).sum()))/(len(y_ungk_test)+len(y_gk_test))
print('xgb模型评价 MAE:\n',xgb_MAE)

# 观察拟合情况
score = r2_score(xgb_pred_gk,y_gk_test)
print(score)
plt.scatter(y_gk_test,xgb_pred_gk)
plt.plot([y_gk_test.min(),y_gk_test.max()],[y_gk_test.min(),y_gk_test.max()],color = 'red',linestyle = "--")
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('xgb_gk')
plt.show()

score = r2_score(xgb_pred_ungk,y_ungk_test)
print(score)
plt.scatter(y_ungk_test,xgb_pred_ungk)
plt.plot([y_ungk_test.min(),y_ungk_test.max()],[y_ungk_test.min(),y_ungk_test.max()],color = 'red',linestyle = "--")
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('xgb_ungk')
plt.show()


# 变量重要性
plot_importance(xgb_gk)
plt.show()

plot_importance(xgb_ungk)
plt.show()


# max_depth=7, learning_rate=0.1, n_estimators=150      10
# 'learning_rate': 0.1, 'max_depth': 2, 'n_estimators' 185   5

grid_dicateg = GridSearchCV(estimator = xgb.XGBRegressor(),param_grid=paramers,cv = 5)
grid_dicateg.fit(X_ungk_train,y_ungk_train)
grid_dicateg.best_params_
xgb1_cv5 = grid_dicateg.predict(X_ungk_train)
xgb_MAE_ungk_cv5 = mean_absolute_error(y_ungk_train,xgb1_cv5)
print('xgb模型评价 MAE_gk:\n',xgb_MAE_ungk_cv5)
 # 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 175
 # 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 185