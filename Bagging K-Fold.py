import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve
import pickle
from matplotlib import pyplot

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

parameters = {
    'n_estimators':(11,19,29,39),
    'oob_score':(False,True)
}
#
# myTree = pickle.load(open("BestTree","rb")) #On récupère le meilleur arbre que nous avions trouvé
# myBag = BaggingClassifier(myTree)
# Grid = GridSearchCV(myBag,parameters,cv=3,verbose=1)
# print("GridSearch launched")
# Grid.fit(x_train,y_train.iloc[:,1].values)
# myBag = Grid.best_estimator_
# pickle.dump(myBag, open("BestBag","wb"))

myBag = pickle.load(open("BestBag","rb"))
print("Best Bagging selected is : ")
print(myBag.get_params)
print("Accuracy on training set : "  + str(myBag.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(myBag.score(x_valid,y_valid.iloc[:,1].values)))
y_pred = myBag.predict(x_valid)


name = 'Bagging'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred,pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()