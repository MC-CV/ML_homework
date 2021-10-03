import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


x_train = pd.read_csv(r'data1forEx\train1_icu_data.csv')
y_train = pd.read_csv(r'data1forEx\train1_icu_label.csv')
x_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

score = []
training_accuracy = []
fit_time = []
score_time = []
for i in range(1,10):   
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    training_accuracy.append(knn.score(x_train,y_train))
    score.append(knn.score(x_test,y_test))
    fit_time.append(np.mean(cross_validate(knn, x_train, y_train, cv=5)['fit_time']))
    score_time.append(np.mean(cross_validate(knn, x_train, y_train, cv=5)['score_time']))  
k = [i for i in range(1,10)]
plt.plot(k,score,label="test_score", color='b')
plt.plot(k,training_accuracy,label="train_score", color='r')
plt.xlabel("k")
plt.ylabel("score")
plt.legend()
plt.show()

plt.plot(k,fit_time,label="fit_time", color='r')
plt.plot(k,score_time,label="score_time", color='b')
plt.xlabel("k")
plt.ylabel("time")
plt.legend()
plt.show()
