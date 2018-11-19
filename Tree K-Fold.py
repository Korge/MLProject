import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import roc_curve
from matplotlib import pyplot


x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

parameters = {
    'max_depth':(2,3,6,10,13),
    'min_samples_split':(2,5,10,15,25,40,60)
}
# myTree = DecisionTreeClassifier()
# Grid = GridSearchCV(myTree,parameters,cv=5)
# Grid.fit(x_train,y_train)
# myTree = Grid.best_estimator_
# pickle.dump(myTree, open("BestTree","wb"))

myTree = pickle.load(open("BestTree","rb"))
print("Best Tree selected is : ")
print(myTree.get_params)
print("Accuracy on training set : "  + str(myTree.score(x_train,y_train)))
print("Accuracy on valid set : "  + str(myTree.score(x_valid,y_valid)))
y_pred = myTree.predict(x_valid)


name = 'Arbre de Classification'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred[:,1],pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()