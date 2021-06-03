# 载入包
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols #加载ols模型
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
from sklearn.metrics import mean_absolute_error


# 导入数据
train = pd.read_csv(r'D:\xzy\数据\data_onehot.csv')
train = train.drop('Unnamed: 0',axis = 1)
train['def1'] =train['def'] 
train = train.drop('def',axis = 1 )
# 拆分训练集
gk_train = (train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)).dropna(axis = 0)
gk_train = gk_train.drop(['gk'],axis = 1)

ungk_train = (train.drop(['gk'],axis = 1)).dropna(axis = 0)
ungk_train = ungk_train.drop(['rw','rb','st','lw','cf','cam','cm','cdm','cb','lb'],axis = 1)

# 拆分训练集
X_gk,y_gk = gk_train,gk_train['y']
X_gk_train,X_gk_test,y_gk_train,y_gk_test = train_test_split(X_gk,y_gk,test_size = 0.2,random_state =100)

X_ungk,y_ungk = ungk_train,ungk_train['y']
X_ungk_train,X_ungk_test,y_ungk_train,y_ungk_test = train_test_split(X_ungk,y_ungk,test_size = 0.2,random_state =100)
 


#定义向前逐步回归函数
def forward_select(data,target):
    variate=set(data.columns)  #将字段名转换成字典类型
    variate.remove(target)  #去掉因变量的字段名
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    #循环筛选变量
    while variate:
        aic_with_variate=[]
        for candidate in variate:  #逐个遍历自变量
            formula="{}~{}".format(target,"+".join(selected+[candidate]))  #将自变量名连接起来
            aic=ols(formula=formula,data=data).fit().aic  #利用ols训练模型得出aic值
            aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  #降序排序aic值
        best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        if current_score>best_new_score:  #如果目前的aic值大于最好的aic值
            variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
            selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
            current_score=best_new_score  #最新的分数等于最好的分数
            print("aic is {},continuing!".format(current_score))  #输出最小的aic值
        else:
            print("for selection over!")
            break
    formula="{}~{}".format(target,"+".join(selected))  #最终的模型式子
    print("final formula is {}".format(formula))
    model=ols(formula=formula,data=data).fit()
    return(model)

select1 = forward_select(data = X_gk_train,target = "y")
model_gk=ols(" y~international_reputation+potential+dri+age+natinality_34+club_604+gk_handling+club_517+club_11+reactions+club_10+natinality_63+club_163+club_252+club_12+club_203+club_555+gk_diving+gk_positioning+club_412+natinality_1+club_309+club_323+club_621+club_49+club_221+league_20+club_27+club_478+club_398+club_68+club_57+short_passing+club_586+league_23+club_245+club_642+club_366+club_495+club_413+club_608+club_153+club_389+club_85+club_388+club_622+club_148+club_13+club_63+club_581+club_500+club_158+vision+club_460+club_504+club_339+club_25+club_595+natinality_86+club_351+natinality_8+club_524+gk_kicking+club_302+club_327+club_360+club_322+natinality_16+natinality_12+club_538+club_116+club_311+club_404+natinality_70+club_42+club_337+finishing+natinality_65+preferred_foot+natinality_68+club_105+club_151+club_224+club_492+marking+club_548+league_5+club_546+club_182+league_30+league_17+club_577+club_364+league_7+natinality_132+natinality_49+league_39+ball_control+sprint_speed+agility+club_331+club_214+club_373+natinality_33+league_10+club_540",data=X_gk_train).fit()
print(model_gk.summary())
pre_gk = model_gk.predict(X_gk_test)
MAE_gk = mean_absolute_error(y_gk_test,pre_gk)
print(MAE_gk)
#128.9643528659493

select2 = forward_select(data = X_ungk_train,target = "y")    
model_ungk=ols(" y~international_reputation+potential+reactions+club_622+league_9+skill_moves+stamina+club_604+club_347+league_40+club_103+club_265+club_227+club_323+club_621+club_412+league_13+club_581+league_10+club_601+age+strength+finishing+natinality_54+club_379+club_85+club_163+club_11+club_351+club_500+club_517+natinality_111+club_276+short_passing+sho+pac+dribbling+club_63+club_605+club_109+league_5+club_485+club_311+club_19+club_600+club_253+club_626+club_158+league_31+league_6+natinality_81+club_257+natinality_147+club_68+club_131+club_594+club_72+club_640+natinality_112+club_483+natinality_149+club_597+club_132+club_364+natinality_14+club_261+club_57+club_555+league_25+shot_power+vision+club_151+club_340+natinality_6+natinality_68+natinality_107+club_110+club_616+club_342+club_309+club_25+club_487+club_497+natinality_133+club_218+club_628+work_rate_def+club_10+club_206+weak_foot+club_346+natinality_136+league_41+natinality_164+club_185+natinality_106+natinality_109+club_404+club_524+club_128+club_506+club_44+club_401+club_471+club_578+club_215+club_145+natinality_116+marking+def1+long_passing+club_439+natinality_161+penalties+club_328+natinality_51+BMI+club_373+natinality_102+club_243+heading_accuracy+club_457+league_14+ball_control+club_376+natinality_57+club_383+club_9+natinality_84+club_288+club_429+club_169+club_252+club_320+natinality_103+club_154+club_566+club_542+league_12+club_447+club_81+club_41+club_82+league_26+league_35+club_291+club_636+club_214+club_478+league_34+natinality_100+club_119+club_104+club_505+natinality_155+league_29+natinality_108+club_212+natinality_119+club_161+club_574+club_270+club_482+club_138",data=X_ungk_train).fit()
model_ungk.summary()
pre_ungk = model_ungk.predict(X_ungk_test)
MAE_ungk = mean_absolute_error(y_ungk_test,pre_ungk)
print(MAE_ungk)
#134.32514234806237



xgb_MAE = (((abs(y_gk_test - pre_gk )).sum())+((abs(y_ungk_test - pre_ungk)).sum()))/(len(y_ungk_test)+len(y_gk_test))
print('xgb模型评价 MAE:\n',xgb_MAE)