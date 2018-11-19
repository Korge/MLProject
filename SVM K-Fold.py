import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import numpy as np

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

parameters = {
    'kernel':("linear","poly","rbf","sigmoid"),
    'gamma':[1e-1,3,5,10],
    'C':[1e-1,1e0,1e1,1e2]
}
#
# mySVM = SVC(max_iter=1500)
# Grid = GridSearchCV(mySVM,parameters,cv=3,verbose=1)
# #Dans le cas de SVM nous devons n'avoir qu'une colonne de label, représentant la classe
# print("GridSearch launched")
# Grid.fit(x_train,y_train.iloc[:,1].values) #Le label prédit sera : "Le salaire est-il supérieur à 50K ?"  oui -> 1 sinon 0
# mySVM = Grid.best_estimator_
# pickle.dump(mySVM, open("BestSVM","wb"))

mySVM = pickle.load(open("BestSVM","rb"))
print("Best SVM selected is : ")
print(mySVM.get_params)
print("Accuracy on training set : "  + str(mySVM.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(mySVM.score(x_valid,y_valid.iloc[:,1].values)))
# plotRoc(mySVM,x_train,y_train.iloc[:,1],False)

y_pred = mySVM.predict(x_valid)
name = 'SVM'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred,pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()