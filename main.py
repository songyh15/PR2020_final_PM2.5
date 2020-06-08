import pandas as pd
import numpy as np
import os 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

def data_generate(data_pm25_filled):
    data = []
    label = []
    data_pm25_filled_array = data_pm25_filled.iloc[:,3:].values
    for i in range(data_pm25_filled_array.shape[0] -12 -6):
        data_now = data_pm25_filled_array[i:i+12,:]
        label_now = data_pm25_filled_array[i+12:i+18,:]
        data.append(data_now)
        label.append(label_now)
    data = np.array(data)
    label = np.array(label)
    return data, label

def Evaluation(label, predict):
    MAE = np.mean(np.abs(label - predict))
    RMSE = np.power(np.mean(np.power(label - predict,2)) ,0.5)

    label_grade = label
    label_grade[label_grade < 35] = 1
    label_grade[label_grade > 250] = 6
    label_grade[label_grade > 150] = 5
    label_grade[label_grade > 115] = 4
    label_grade[label_grade > 75] = 3
    label_grade[label_grade > 35] = 2
    
    predict_grade = predict
    predict_grade[predict_grade < 35] = 1
    predict_grade[predict_grade > 250] = 6
    predict_grade[predict_grade > 150] = 5
    predict_grade[predict_grade > 115] = 4
    predict_grade[predict_grade > 75] = 3
    predict_grade[predict_grade > 35] = 2
    
    res = np.zeros(label_grade.shape)
    res[label_grade == predict_grade] = 1
    num_cor = res.sum()
    num_all = res.shape[0] * res.shape[1]
    prec = num_cor/num_all
    return MAE,RMSE,prec

# 读取数据
# 注: 20141231读取不了, 删了; 20151230,1231是空文件, 删了

# data_folder = os.walk(r"data")  
# for path,dir_list,file_list in data_folder:  
#     for file_name in file_list:  
#         path_now = os.path.join(path, file_name)
#         if file_name == "beijing_all_20150101.csv" :
#             data_all = pd.read_csv(path_now)
#             print(path_now)
#         elif file_name[:11] == "beijing_all" :
#             data_now = pd.read_csv(path_now)
#             data_all = pd.concat([data_all, data_now], axis=0)
#             print(path_now)
# data_all.to_csv("data_all.csv")
         
data_all = pd.read_csv("data_all.csv")
# 选取pm2.5数据
data_all_pm25 = data_all[data_all['type']=='PM2.5']
# 删去无用的前三列
data_all_pm25 = data_all_pm25.iloc[:,3:]
# "就近"填充缺失数据
data_all_pm25_filled = data_all_pm25.fillna(method='ffill')
data_all_pm25_filled = data_all_pm25_filled.fillna(method='bfill')
# 划分数据集
data_train_pm25_filled = data_all_pm25_filled[data_all_pm25_filled['date'] < 20200000]
data_train_pm25_filled = data_train_pm25_filled[data_train_pm25_filled['date'] > 20150000]
data_val_pm25_filled = data_all_pm25_filled[data_all_pm25_filled['date'] > 20200000]
data_test_pm25_filled = data_all_pm25_filled[data_all_pm25_filled['date'] < 20150000]
# 制作时序数据集
train_data,train_label = data_generate(data_train_pm25_filled)
val_data,val_label = data_generate(data_val_pm25_filled)
test_data,test_label = data_generate(data_test_pm25_filled)

# 训练并测试模型
models = [DecisionTreeRegressor(),RandomForestRegressor(),AdaBoostRegressor,GradientBoostingRegressor]
train_predict_all = []
val_predict_all = []
test_predict_all = []
for model_now in models:
    train_predict = np.zeros(train_label.shape)
    val_predict = np.zeros(val_label.shape)
    test_predict = np.zeros(test_label.shape)
    
    for ti in range(train_label.shape[1]):
        for lo in range(train_label.shape[2]):
            model_now.fit(train_data[:,:,lo], train_label[:,ti,lo])
            train_predict[:,ti,lo] = model_now.predict(train_data[:,:,lo])
            val_predict[:,ti,lo] = model_now.predict(val_data[:,:,lo])
            test_predict[:,ti,lo] = model_now.predict(test_data[:,:,lo])
            
    train_predict_all.append(train_predict)
    val_predict_all.append(val_predict)
    test_predict_all.append(test_predict)

# 评价模型性能

for m in range(4):
    print(models[m])
    for i in range(6):
        MAE, RMSE,PREC= Evaluation(val_label[:,i,:], val_predict_all[m][:,i,:])
        print('time:'+str(i+1)+' '+'MAE = '+str(MAE)+' '+'RMSE = '+str(RMSE)+' '+'PREC = '+str(PREC))
