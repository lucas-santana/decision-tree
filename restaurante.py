#!/usr/bin/python

import numpy as np
from sklearn import tree

# 0 - Alternate: há um restaurante alternativo na redondeza? 
# 1 - Bar: existe um bar confortável onde se esperar?
# 2 - Fri/Sat: hoje é sexta ou sábado ?
# 3 - Hungry: estou com fome?
# 4 - Patrons: numero de pessoas no restaurante (0 - None, 1 - Some, 2 - Full)
# 5 - Price: faixa de preços (0- $, 1 - $$, 2 - $$$)
# 6 - Raining: está a chover?
# 7 - Reservation: temos reserva?
# 8 - Type: tipo do restaurante (0 - French, 1 - Italian, 2 - Thai, 3 - Burger)
# 9 - WaitEstimate: tempo de espera estimado (0 => 0-10, 1 => 10-30, 2 => 30-60, 3 => >60)

X = np.array([
#                0 1 2 3 4 5 6 7 8 9       
                [1,0,0,1,1,2,0,1,0,0], #X1
                [1,0,0,1,2,0,0,0,2,2], #X2
                [0,1,0,0,1,0,0,0,3,0], #X3
                [1,0,1,1,2,0,0,0,2,1], #X4
                [1,0,1,0,2,2,0,1,0,3], #X5
                [0,1,0,1,1,1,1,1,1,0], #X6
                [0,1,0,0,0,0,1,0,3,0], #X7
                [0,0,0,1,1,1,1,1,2,0], #X8
                [0,1,1,0,2,0,1,0,3,3], #X9
                [1,1,1,1,2,2,0,1,1,1], #X10
                [0,0,0,0,0,0,0,0,2,0], #X11
                [1,1,1,1,2,0,0,0,3,2]  #X12
            ])
Y = np.array([1,0,1,1,0,1,0,1,0,0,0,1])

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

res = clf.predict([[1,0,1,1,0,1,0,1,0,0]])
print (res[0])