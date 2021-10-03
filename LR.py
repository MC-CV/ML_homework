import numpy as np
from numpy.core.defchararray import title
from numpy.lib.function_base import iterable
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score,confusion_matrix
from visdom import Visdom
import numpy as np
import math
import time
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import seaborn


def label_process(label):
    for i in range(len(label)):
        if int(label[i]) == 0:
            label[i] = -1
    return label

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class lr():
    def __init__(self,X,Y,lr,num_arg):
        self.X = np.mat(X)
        self.Y = np.mat(Y)
        self.rate = lr
        self.final_w = 0
        self.num_arg = num_arg
    def train(self):
        w = np.ones((self.num_arg,1))
        iterater = 0
        while iterater<=1000:
            h = sigmoid(self.X * w)
            # import pdb;pdb.set_trace()
            w += self.rate*self.X.T*(self.Y - h) # 109,1
            iterater += 1
        self.final_w = w
        # print(self.final_w)
    def pred(self,x_test):
        x_test = np.mat(x_test)
        return sigmoid(x_test * self.final_w)
    def eval(self,y_pred,y_true):
        y_true = np.array(y_true)
        num = 0
        for i in range(len(y_pred)):
            # import pdb;pdb.set_trace()
            if int(y_pred[i]>=0.5) == y_true[i]:
                num += 1
        print('score:',num/len(y_true))
        print('error rates:',1-num/len(y_true))
        




if __name__ == '__main__':
    x_train = pd.read_csv(r'data1forEx\train1_icu_data.csv')
    y_train = pd.read_csv(r'data1forEx\train1_icu_label.csv')
    x_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
    y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train = np.insert(x_train, 0, 1, axis=1)
    x_test = np.insert(x_test, 0, 1, axis=1)

    normalizer=Normalizer(norm='l2')
    x_train = normalizer.transform(x_train)
    x_train = x_train[:,:]
    x_test = normalizer.transform(x_test)
    x_test = x_test[:,:]


    # logr = lr(x_train,y_train,0.001,x_train.shape[1])
    # logr.train()
    # y_pred = logr.pred(x_test)
    # logr.eval(y_pred,y_test)

    
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # plt.plot(fpr, tpr, label='ROC')  
    # plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
    # plt.title('ROC')  
    # plt.xlabel('false positive rate') 
    # plt.ylabel('true positive rate')  
    # plt.legend(loc=0)
    # plt.show()

    # kf = KFold(n_splits=5)
    # i = 1
    # for train_index, test_index in kf.split(x_train,y_train):
    #     X_train, X_test = x_train[train_index], x_train[test_index]
    #     Y_train, Y_test = y_train[train_index], y_train[test_index]
    #     logr = lr(X_train,Y_train,0.001,X_train.shape[1])
    #     logr.train()
    #     y_pred = logr.pred(X_test)
    #     print('第',i,'次:')
    #     logr.eval(y_pred,Y_test)
    #     i += 1
    x_train = pd.read_csv(r'data1forEx\train1_icu_data.csv')
    y_train = pd.read_csv(r'data1forEx\train1_icu_label.csv')
    x_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
    y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')

    x_train = x_train.iloc[:,9:12]
    x_train = pd.concat([x_train,y_train],axis=1)
    df_corr = x_train.corr()
    seaborn.heatmap(df_corr, center=0, annot=True, cmap='YlGnBu')
    seaborn.pairplot(x_train)
    plt.show()