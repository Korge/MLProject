import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
import pickle
from matplotlib import pyplot

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

parameters = {
    'n_estimators':(45,50,60,75,90,110)
}
#

# myBoost = AdaBoostClassifier() #Par défaut, un arbre de profondeur 1 sera utilisé (éq. perceptron)
# Grid = GridSearchCV(myBoost,parameters,cv=3,verbose=1)
# print("GridSearch launched")
# Grid.fit(x_train,y_train.iloc[:,1].values)
# myBoost = Grid.best_estimator_
# pickle.dump(myBoost, open("BestBoost","wb"))

myBoost = pickle.load(open("BestBoost","rb"))
print("Best AdaBoostClassifier selected is : ")
print(myBoost.get_params)
print("Accuracy on training set : "  + str(myBoost.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(myBoost.score(x_valid,y_valid.iloc[:,1].values)))

y_pred = myBoost.predict(x_valid)


name = 'Boosting'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred,pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()