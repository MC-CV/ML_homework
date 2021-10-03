import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# 计算协方差和矩阵和平均向量
def cal_cov_and_avg(samples):
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t * t.reshape(108, 1)
    return cov_m, u1

# fisher算法实现
def fisher(c_1, c_2):
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)  # 奇异值分解
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)

def inference(sample, w, c_1, c_2):
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return int(abs(pos - center_1) < abs(pos - center_2))

def test(w,x_test,c_1,c_2):
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    y_pred = np.zeros([x_test.shape[0],1])
    for idx,i in enumerate(x_test):
        pos = np.dot(w.T, i)
        y_pred[idx] = 1 if abs(pos - center_1) < abs(pos - center_2) else 0
    return y_pred

def eval(y_pred,y_test):
    num = 0
    for idx,i in enumerate(y_pred):
        if i == y_test[idx]:
            num += 1
    print('错误率：',(len(y_pred)-num)/len(y_pred))

if __name__ == '__main__':
    x_train = pd.read_csv(r'data1forEx\train1_icu_data.csv')
    y_train = pd.read_csv(r'data1forEx\train1_icu_label.csv')
    x_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
    y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    index1 = np.array([index for (index, value) in enumerate(y_train) if value == 1])  # 获取类别1的indexs
    index2 = np.array([index for (index, value) in enumerate(y_train) if value == 0])  # 获取类别2的indexs
    
    c_1 = x_train[index1]   
    c_2 = x_train[index2]  

    w = fisher(c_1, c_2)  # 调用函数，得到参数w
    out = inference(x_test[0], w, c_1, c_2)   # inference单个样本的结果
    y_pred = test(w,x_test,c_1,c_2) # 对整个测试集inference
    eval(y_pred,y_test) # 评估指标