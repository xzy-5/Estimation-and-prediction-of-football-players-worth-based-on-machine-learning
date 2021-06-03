import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import OneHotEncoder

# 一.读取数据
test = pd.read_csv(r'D:\xzy\数据\data\test.csv',encoding = 'utf8')
train = pd.read_csv(r'D:\xzy\数据\data\train.csv',encoding = 'utf8')
submit = pd.read_csv(r'D:\xzy\数据\data\sample_submit.csv',encoding = 'utf8')

print('test数据集行数:\n',test.shape)
print('train数据集行数:\n',train.shape)
print('submit数据集行数:\n',submit.shape)

print('train数据缺失情况:\n',train.isnull().sum())
print('test数据缺失情况:\n',test.isnull().sum())
pd.set_option("display.max_rows", 1000)#可显示1000行
pd.set_option("display.max_columns", 1000)#可显示1000列

# 二.日期数据转换，计算运动员年龄
curr_date = datetime.datetime.now()
train['birth_date'] = pd.to_datetime(train.birth_date)
train_age = (curr_date-train.birth_date).apply(lambda x:x.days)/365
train.insert(4,'age',train_age)
train = train.drop(['birth_date','id'],axis = 1)

curr_date = datetime.datetime.now()
test['birth_date'] = pd.to_datetime(test.birth_date)
test_age = (curr_date-test.birth_date).apply(lambda x:x.days)/365
test.insert(4,'age',test_age)
test = test.drop(['birth_date','id'],axis = 1)

# 计算BMI指数
train_BMI = 10000.*train['weight_kg'] / (train['height_cm'] ** 2)
train.insert(3,'BMI',train_BMI)
train = train.drop(['weight_kg','height_cm'],axis = 1)

test_BMI = 10000.*test['weight_kg'] / (test['height_cm'] ** 2)
test.insert(3,'BMI',test_BMI)
test = test.drop(['weight_kg','height_cm'],axis = 1)

# 三.对离散数据进行哑变量处理
Lable_maping = {'Low':0,'Medium':1,'High':2}
train['work_rate_att'] = train['work_rate_att'].map(Lable_maping )
train['work_rate_def'] = train['work_rate_def'].map(Lable_maping )

test['work_rate_att'] = test['work_rate_att'].map(Lable_maping )
test['work_rate_def'] = test['work_rate_def'].map(Lable_maping )


# 四.one-hot
one_hot=OneHotEncoder()
data_temp1=pd.DataFrame(one_hot.fit_transform(train[['club']]).toarray(),columns = one_hot.get_feature_names(['club']),dtype='int32')

data_temp2=pd.DataFrame(one_hot.fit_transform(train[['league']]).toarray(),
columns=one_hot.get_feature_names(['league']),dtype='int32')

data_temp3=pd.DataFrame(one_hot.fit_transform(train[['nationality']]).toarray(),
columns=one_hot.get_feature_names(['natinality']),dtype='int32')

data_onehot=pd.concat((train,data_temp1,data_temp2,data_temp3),axis=1)  
data_onehot =data_onehot.drop(['club','league','nationality'],axis = 1)
data_onehot.to_csv(r'D:\xzy\数据\data_onehot.csv')




import time
start = time.clock()
end = time.clock()
print (str(end-start))