########################################
# Introduction to Machine Learning HW6 # 
# Due : 20181205                       #
# 107064522                            #
# Part III: Perceptron                 #
########################################

import numpy as np
import pandas as pd
from math import sqrt


## Data processing _ Training data
# Read Data
tr_data = pd.read_csv('training-data.txt', sep = ",", header = None)
columns = ["at1", "at2", "at3", "at4", "label"]
tr_data.columns = columns

# Label Iris-setosa as '0', Iris-versicolor as '1'
tr = tr_data.drop("label", axis = 1)
tr['label'] = 0
tr.loc[45:90, ['label']] = 1
tr_wol = tr.drop("label", axis = 1)

## Data processing _ Test data
# Read Data
te_data = pd.read_csv('testing-data.txt', sep = ",", header = None).iloc[0:10]
te_data.columns = columns

# Label Iris-setosa as '0', Iris-versicolor as '1'
te = te_data.drop("label", axis = 1)
te['label'] = 0
te.loc[5:10, ['label']] = 1
te_wol = te.drop("label", axis = 1)

## Perceptron
eta = 0.2
epoch = 1
w = np.full((5), 0.2)   # Initial weight
tr_wol['at0'] = 1

hold = 1
count = 0
nerr = 0
while hold:
    err = np.full((len(tr)), 0)   # c(x) - h(x)
    h = np.full((len(tr)), -1)   # Class return by classifier
    err_temp = 0
    print("##### epoch = ", epoch, " #####")
    print(w)
    for i in range(len(tr)):
        if (tr_wol.iloc[i]*w).sum() > 1e-10:
            h[i] = 1
        else:
            h[i] = 0
        err[i] = (tr.iloc[i])['label'] - h[i]
        w = w+eta*err[i]*tr_wol.iloc[i]
    nerr = (err != 0).sum()
    print("Number of errors: ", nerr)
    if nerr != 0:
        epoch += 1
        if (nerr <= 2) & (abs(err_temp-nerr) <= 2):    # accuracy > 97%
            count += 1
        else:
            count = 0
        if count == 10:
            hold = 0
        err_temp = nerr
    else:
        hold = 0

print("##### DONE! #####\n")
print("Number of epoch: ", epoch)
print("Final Weight:\n", w)

## Testing
true = 0
false = 0

for i in range(len(te)):
    if (te_wol.iloc[i]*w).sum() > 1e-10:
        h[i] = 1
    else:
        h[i] = 0
    err[i] = (tr.iloc[i])['label'] - h[i]

    # Caculating accuracy
    if h[i] == te["label"].iloc[i]:
        true += 1
        print("Example ",i+1, ": hit")
    else:
        print("Example ",i+1, ": miss")
        false += 1

accuracy = true/len(te)
print("\nAccuracy = ", accuracy)