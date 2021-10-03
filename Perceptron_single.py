from typing import Iterator
import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from visdom import Visdom
import numpy as np
import time
from sklearn.preprocessing import Normalizer

def label_process(label):
    for i in range(len(label)):
        if int(label[i]) == 0:
            label[i] = -1
    return label

def data_process(l1):
    for i in l1.keys():
        total_len_half = (max(l1[i])-min(l1[i]))/2
        avg = (max(l1[i])+min(l1[i]))/2
        for j in range(len(l1[i])):
            l1[i][j] = int((l1[i][j]-avg)/total_len_half)
    return l1

class Perception(object):
    def __init__(self,X,Y,lr,num_arg):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.rate = lr
        self.final_w = 0
        self.final_b = 0
        self.num_arg = num_arg
        self.epoch = 10
    def train(self):
        num = [i for i in range(len(self.Y))]
        w = np.ones(self.num_arg)
        b = 0
        iterater = 0
        # for j in range(self.epoch):
        temp = 0
        while iterater<=100000:
            loss = 0
            acc = 0
            for i in num:
                flag = self.Y[i][0] * (np.dot(self.X[i].T,w)+b)
                import pdb;pdb.set_trace()
                if flag <= 0:
                    # import pdb;pdb.set_trace()
                    w += self.rate*self.Y[i]*self.X[i]
                    b += self.rate*self.Y[i]
                    loss -= flag
                    # print(abs(temp - loss)/loss)
                    # if loss!=0 and abs(temp - loss)/loss < 0.00001:
                    #     print(iterater)
                    # temp = loss
                else:
                    acc += 1
                if acc >= len(self.Y):
                    break
                iterater += 1
                
            viz.line([acc/len(num)], [iterater], win='accuracy', update='append')
            viz.line([loss/len(num)], [iterater], win='train_loss', update='append')
            print('iterater',iterater,'loss:',loss/len(num))

        self.final_w = w
        self.final_b = b
        print(self.final_w,self.final_b)
    def test(self,X):
        self.pred = np.zeros(len(X))
        for i in range(len(X)):
            # self.pred[i] = np.sign(np.inner(self.final_w,X[i])+self.final_b)
            self.pred[i] = np.sign(np.dot(self.X[i].T,self.final_w)+self.final_b)
        return self.pred
    def eval(self,y_pred,y_true):
        y_true = np.array(y_true)
        num = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                num += 1
        print('score:',num/len(y_true))
        print('error rates:',1-num/len(y_true))


if __name__ == '__main__':
    viz = Visdom()
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0.], [0], win='accuracy', opts=dict(title='accuracy'))
    x_train1 = pd.read_csv(r'data1forEx\train1_icu_data.csv')
    y_train1 = pd.read_csv(r'data1forEx\train1_icu_label.csv')
    x_train2 = pd.read_csv(r'data1forEx\train2_icu_data.csv')
    y_train2 = pd.read_csv(r'data1forEx\train2_icu_label.csv')
    x_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
    y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')
    # x_test = pd.read_csv(r'data1forEx\test2_icu_data.csv')
    # y_test = pd.read_csv(r'data1forEx\test2_icu_label.csv')
    x_train = np.vstack([x_train1,x_train2])
    y_train = np.vstack([y_train1,y_train2])
    x_train1 = np.array(x_train1)
    y_train1 = np.array(y_train1)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    normalizer=Normalizer(norm='l2')

    x_train = normalizer.transform(x_train1)
    x_train = x_train[:,:]
    x_test = normalizer.transform(x_test)
    x_test = x_test[:,:]

    y_proce_train = label_process(y_train1)
    y_proce_test = label_process(y_test)
    p1 = Perception(x_train,y_proce_train,0.01,x_train.shape[1])
    p1.train()
    y_pred = p1.test(x_test)
    p1.eval(y_pred,y_proce_test)