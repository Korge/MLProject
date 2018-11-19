import pandas as pd
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from PlotRoc import plotRoc

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')


parameters = {'weights':('uniform', 'distance'),
              'metric': ('minkowski','euclidean','manhattan','chebyshev'),
              'p': [1,2],
              'n_neighbors':[1,3,5]}

# myKNN = KNeighborsClassifier()
# Grid = GridSearchCV(myKNN,parameters,cv=3,verbose=3)
# Grid.fit(x_train,y_train)
# myTree = Grid.best_estimator_
# pickle.dump(myTree, open("BestKNN","wb"))

myKNN = pickle.load(open("BestKNN","rb"))
print("Best KNN selected is : ")
print(myKNN.get_params)
print("Accuracy on training set : "  + str(myKNN.score(x_train,y_train)))
print("Accuracy on valid set : "  + str(myKNN.score(x_valid,y_valid)))
y_pred = myKNN.predict(x_valid)


name = 'Plus proche voisin'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred[:,1],pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()