########################################
# Introduction to Machine Learning HW6 # 
# Due : 20181205                       #
# 107064522                            #
# Part II: Adaboost by inventor        #
########################################

import numpy as np
import pandas as pd
from math import sqrt

## Implement knn
# Calculate the Euclidean Distance
def distance(row, x):
    dist = 0
    for i, j in x.items():
        dist += (row[i] - j)**2
    dist = sqrt(dist)
    return dist

# knn
def knn(tr_data, k, x):
    Euclidean_dist = []
    #calculate the distance between x and each data
    for index, row in tr_data.iterrows():
        Euclidean_dist.append((index, distance(row, x)))
    #sort the distance
    Euclidean_dist = sorted(Euclidean_dist, key = lambda dist:dist[1])
    #pick the k-nearest neighbors
    knn_idx = [index[0] for index in Euclidean_dist[0:k]]
    knn = tr_data.iloc[knn_idx]
    return knn['label'].value_counts().idxmax()


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

## Adaboost
# Initialize examples distribution : Uniform
p = np.ones(len(tr_data))/len(tr_data)    # Initial distribution
T = list()    # 9 tr. subset with 10 examples/subset
alpha = np.zeros(9)
for subset in range(9):
    tr['output'] = -1
    tr['e'] = -1

    temp_idx = np.sort(np.random.choice(len(tr), 10, p = p))
    temp_T = pd.DataFrame(tr.iloc[temp_idx]).reset_index().drop("index", axis = 1)
    T.append(temp_T[['at1', 'at2', 'at3', 'at4', 'label']])

    for i in range(len(tr)):
        output = knn(temp_T, 1, tr_wol.iloc[i])
        tr['output'][i] = output
        if (tr.iloc[i])['output'] == (tr.iloc[i])['label']:
            tr['e'][i] = 0  # True
        else:
            tr['e'][i] = 1  # False

    # Update the distribution
    epsilon = (tr['e']*p).sum()
    beta = epsilon/(1-epsilon)

    if beta != 0:
        for j in range(len(tr)):
            if tr['e'][j] == 0: # True
                p[j] = p[j]*sqrt(beta)    # Update examples distribution
            else:
                p[j] = p[j]/sqrt(beta)    # Update examples distribution
                

        p = p/p.sum()   # Normalization

    
    # Voting weight
    if epsilon != 0:
        alpha[subset] = np.log((1-epsilon)/epsilon)/2
    else:
        alpha[subset] = alpha[subset-1]

print("##### DONE! #####\n")
print("Final Weight: ", alpha)

## Testing
true = 0
false = 0

for i in range(len(te)):
    h = np.full((len(T)), -1)   # For storing output of subclassifiers
    H = -1  # Master classifier
    for j in range(len(T)):
        h[j] = knn(T[j], 1, te_wol.iloc[i])
    print(h)
    # Weight majority voting
    if (alpha[h == 1]).sum() > (alpha[h == 0]).sum():
        H = 1
    else:
        H = 0
    
    # Caculating accuracy
    if H == te["label"].iloc[i]:
        true += 1
        print("Example ",i+1, ": hit")
    else:
        print("Example ",i+1, ": miss")
        false += 1

accuracy = true/len(te)
print("Accuracy = ", accuracy)