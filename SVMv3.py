import pandas as pd
url = "data_v3.csv"
insider = pd.read_csv(url, header=0)
print insider.shape

row_num = insider['side'].count()+1
train_num = int(row_num /3*2)
test_num = -1*int(row_num /3)

print "Training size: %d, Testing size: %d" % (train_num, test_num)

col_list = ['side', 'return_t5', "return_t30", "vol_sh_out_pct","stake_pct_chg", "tran_value","mkt_cap", "prev_tran_num","hit_rate_5d", "hit_rate_30d", "hit_rate_90d"]

# Apply Min / Max Scaling
def scaler(col_name):
    insider[col_name] = (insider[col_name]-insider[col_name].min())/ (insider[col_name].max()-insider[col_name].min())

for i in col_list:
    scaler(i)

X_train = insider[col_list][:train_num]
y_train_5d = insider.return_5d[:train_num]
y_train_30d = insider.return_30d[:train_num]
y_train_90d = insider.return_90d[:train_num]

X_test = insider[col_list][test_num:]
y_test_5d = insider.return_5d[test_num:]
y_test_30d = insider.return_30d[test_num:]
y_test_90d = insider.return_90d[test_num:]

import numpy as np
from sklearn.svm import SVC
clf_linear = SVC(kernel = "linear")
#clf.fit(X_train, y_train_5d)
#print clf.score(X_test, y_test_5d)
#clf.fit(X_train, y_train_30d)
#print clf.score(X_test, y_test_30d)
clf_linear.fit(X_train, y_train_90d)
print clf_linear.score(X_test, y_test_90d)

clf_poly = SVC(kernel = "poly")
clf_poly.fit(X_train, y_train_90d)
print clf_poly.score(X_test, y_test_90d)
clf_poly.fit(X_train, y_train_5d)
print clf_poly.score(X_test, y_test_5d)
clf_poly.fit(X_train, y_train_30d)
print clf_poly.score(X_test, y_test_30d)


clf_sig = SVC(kernel = "sigmoid")
clf_sig.fit(X_train, y_train_90d)
print clf_sig.score(X_test, y_test_90d)

