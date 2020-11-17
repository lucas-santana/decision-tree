#!/usr/bin/python

import numpy as np
from sklearn import tree

X = np.array([[0,0], [1,1]])
Y = np.array([0,1])

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

res = clf.predict([[1,0]])
print (res[0])