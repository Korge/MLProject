import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from matplotlib import pyplot


x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

#
# knn1 = KNeighborsClassifier(n_neighbors=1, p=1, leaf_size=100, weights='uniform', metric='minkowski')
# knn3 = KNeighborsClassifier(n_neighbors=3, p=1, leaf_size=1500, weights='uniform', metric='minkowski')
# mlp = MLPClassifier(solver='adam',hidden_layer_sizes=(10,10,35), max_iter=350)
# svm = SVC(C=0.1, gamma=0.1, kernel='sigmoid',cache_size=200,degree=3, max_iter=1500)
# tree = DecisionTreeClassifier(max_depth=10, min_samples_split=60)
#
# myVote = VotingClassifier(estimators=[('knn1', knn1),('knn3',knn3), ('mlp',mlp), ('svm', svm), ('tree', tree)], voting='hard')
# myVote.fit(x_train,y_train.iloc[:,1].values)
# pickle.dump(myVote, open("myVote","wb"))

myVote = pickle.load(open("myVote","rb"))


print("Accuracy on training set : "  + str(myVote.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(myVote.score(x_valid,y_valid.iloc[:,1].values)))
y_pred = myVote.predict(x_valid)
name = 'Vote majoritaire'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred,pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()