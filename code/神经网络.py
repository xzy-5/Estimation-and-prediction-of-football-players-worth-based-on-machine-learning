
from tensorflow import keras as k
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler


# 读取数据
# 导入数据
train = pd.read_csv(r'C:\Users\76081\Desktop\data_onehot.csv',encoding = 'utf8')
train = train.drop('Unnamed: 0',axis = 1)

# 根据缺失值，划分守门员和非守门员
gk_train = (train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)).dropna(axis = 0)

ungk_train = (train.drop(['gk'],axis = 1)).dropna(axis = 0)

# 拆分训练集
X_gk,y_gk = gk_train.drop('y',axis = 1),gk_train['y']
X_gk_train,X_gk_test,y_gk_train,y_gk_test = train_test_split(X_gk,y_gk,test_size = 0.2,random_state =100)

X_ungk,y_ungk = ungk_train.drop('y',axis = 1),ungk_train['y']
X_ungk_train,X_ungk_test,y_ungk_train,y_ungk_test = train_test_split(X_ungk,y_ungk,test_size = 0.2,random_state =100)
 

#归一化
std_scaler = StandardScaler()

# 归一化转化
X_gk_train = std_scaler.fit_transform(X_gk_train)
X_gk_test = std_scaler.fit_transform(X_gk_test)


X_ungk_train = std_scaler.fit_transform(X_ungk_train)
X_ungk_test = std_scaler.fit_transform(X_ungk_test)


X_gk_train = np.array(X_gk_train)
y_gk_train = np.array(y_gk_train)
X_gk_test = np.array(X_gk_test)
y_gk_test = np.array(y_gk_test)
X_gk_train = X_gk_train.astype(np.float)
y_gk_train = y_gk_train.astype(np.float)
X_gk_test = X_gk_test.astype(np.float)
y_gk_test = y_gk_test.astype(np.float)

X_ungk_train = np.array(X_ungk_train)
y_ungk_train = np.array(y_ungk_train)
X_ungk_test = np.array(X_ungk_test)
y_ungk_test = np.array(y_ungk_test)
X_ungk_train = X_ungk_train.astype(np.float)
y_ungk_train = y_ungk_train.astype(np.float)
X_ungk_test = X_ungk_test.astype(np.float)
y_ungk_test = y_ungk_test.astype(np.float)




#神经网络训练
#gk
#神经网络训练
#gk
gkmodel = k.models.Sequential()
gkmodel.add(k.layers.Dense(800,activation='sigmoid',input_dim= 885))
gkmodel.add(k.layers.Dense(512,activation='sigmoid'))
gkmodel.add(k.layers.Dense(128,activation='sigmoid'))
gkmodel.add(k.layers.Dense(64,activation='sigmoid'))
gkmodel.add(k.layers.Dense(32,activation='sigmoid'))
gkmodel.add(k.layers.Dense(8,activation='sigmoid'))
gkmodel.add(k.layers.Dense(1))
gkmodel.compile(loss='mean_absolute_error',optimizer='rmsprop',metrics=['mae'])


model_gk = gkmodel.fit(X_gk_train,y_gk_train,epochs=15)
gk_loss = gkmodel.evaluate(X_gk_test,y_gk_test,batch_size=1024)
print('loss_gk:\n',gk_loss)
model_pre_gk = gkmodel.predict(X_gk_test,batch_size=10)
#117

#ungk
ungkmodel = k.models.Sequential()
ungkmodel.add(k.layers.Dense(800,activation='sigmoid',input_dim=894))
gkmodel.add(k.layers.Dense(700,activation='sigmoid'))
gkmodel.add(k.layers.Dense(600,activation='sigmoid'))
gkmodel.add(k.layers.Dense(512,activation='sigmoid'))
gkmodel.add(k.layers.Dense(356,activation='sigmoid'))
gkmodel.add(k.layers.Dense(218,activation='sigmoid'))
gkmodel.add(k.layers.Dense(128,activation='sigmoid'))
gkmodel.add(k.layers.Dense(64,activation='sigmoid'))
gkmodel.add(k.layers.Dense(32,activation='sigmoid'))
gkmodel.add(k.layers.Dense(8,activation='sigmoid'))
gkmodel.add(k.layers.Dense(1))
gkmodel.compile(loss='mean_absolute_error',optimizer='rmsprop',metrics=['mae'])


model_ungk = ungkmodel.fit(X_ungk_train,y_ungk_train,epochs=10)
ungk_loss = ungkmodel.evaluate(X_ungk_test,y_ungk_test,batch_size=64)
print('loss_ungk:\n',ungk_loss)
model_pre_ungk = ungkmodel.predict(X_ungk_test,batch_size=1024)
