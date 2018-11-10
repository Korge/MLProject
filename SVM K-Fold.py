import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import numpy as np

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

# parameters = {
#     'C':[1,5],
#     'kernel':("linear","poly","rbf","sigmoid"),
#     'max_iter':(500,1000)
# }
#
# mySVM = SVC()
# Grid = GridSearchCV(mySVM,parameters,cv=3,verbose=1)
# #Dans le cas de SVM nous devons n'avoir qu'une colonne de label, représentant la classe
# print("GridSearch launched")
# Grid.fit(x_train,y_train.iloc[:,1].values) #Le label prédit sera : "Le salaire est-il supérieur à 50K ?"  oui -> 1 sinon 0
# mySVM = Grid.best_estimator_
# print("Best Tree selected is : ")
# print(mySVM.get_params)
# pickle.dump(mySVM, open("BestSVM","wb"))

mySVM = pickle.load(open("BestSVM","rb"))
print("Accuracy on training set : "  + str(mySVM.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(mySVM.score(x_valid,y_valid.iloc[:,1].values)))