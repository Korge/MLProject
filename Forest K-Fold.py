import pandas as pd
from sklearn.metrics import roc_curve
import pickle
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')



parameters = {
    'n_estimators':(100,300,500),
    'max_depth': (8,11,15)
}
#
# myForest = RandomForestClassifier()
# Grid = GridSearchCV(myForest,parameters,cv=3,verbose=1)
# Grid.fit(x_train,y_train)
# myForest = Grid.best_estimator_
# pickle.dump(myForest, open("BestForest","wb"))

myForest = pickle.load(open("BestForest","rb"))
print("Best Forest selected is : ")
print(myForest.get_params)
print("Accuracy on training set : "  + str(myForest.score(x_train,y_train)))
print("Accuracy on valid set : "  + str(myForest.score(x_valid,y_valid)))
y_pred = myForest.predict(x_valid)


name = 'Forêt Aléatoire'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred[:,1],pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()