import pandas as pd
url = "data_v3.csv"
insider = pd.read_csv(url, header=0)
print insider.shape

row_num = insider['side'].count()+1
train_num = int(row_num /3*2)
test_num = -1*int(row_num /3)

col_list = ['side', 'return_t5', "return_t30", "vol_sh_out_pct","stake_pct_chg", "tran_value","mkt_cap", "prev_tran_num","hit_rate_5d", "hit_rate_30d", "hit_rate_90d"]

X_train = insider[col_list][:train_num]
y_train_5d = insider.return_5d[:train_num]
y_train_30d = insider.return_30d[:train_num]
y_train_90d = insider.return_90d[:train_num]

X_test = insider[col_list][:test_num]
y_test_5d = insider.return_5d[:test_num]
y_test_30d = insider.return_30d[:test_num]
y_test_90d = insider.return_90d[:test_num]


## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(X_train, y_train_5d)
print knn.score(X_test, y_test_5d)
knn.fit(X_train, y_train_30d)
print knn.score(X_test, y_test_30d)
knn.fit(X_train, y_train_90d)
print knn.score(X_test, y_test_90d)


import matplotlib.pyplot as plt  
import numpy as np

def knn_train(n):
    knn2 = KNeighborsClassifier(n_neighbors=n)
    knn2.fit(X_train, y_train_5d)
    return knn2.score(X_test, y_test_5d)

x = np.arange(1, 20, 1)
y = []
for i in x:
    y.append(knn_train(i))

plt.plot(x, np.asarray(y))
plt.show()

