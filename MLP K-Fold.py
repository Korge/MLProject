import pandas as pd
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot


x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')


# parameters = {
#     'hidden_layer_sizes':((10,10,35), (10,10,20)),
#     'max_iter': (250,300,350)
#     }
#
# myMLP =  MLPClassifier()
# Grid = GridSearchCV(myMLP,parameters,cv=3,verbose=3)
# Grid.fit(x_train,y_train)
# myMLP = Grid.best_estimator_
# pickle.dump(myMLP, open("BestMLP","wb"))

myMLP = pickle.load(open("BestMLP","rb"))
print("Best MLP selected is : ")
print(myMLP.get_params)
print("Accuracy on training set : "  + str(myMLP.score(x_train,y_train)))
print("Accuracy on valid set : "  + str(myMLP.score(x_valid,y_valid)))
y_pred = myMLP.predict(x_valid)


name = 'MLP'
fpr, tpr, thr = roc_curve(y_valid.iloc[:,1].values, y_pred[:,1],pos_label=1)
fig = pyplot.figure(figsize=(6, 6))
pyplot.plot(fpr, tpr, '-', lw=2, label=name)
pyplot.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--',label='Courbe théorique')
pyplot.xlabel('False Positive Rate', fontsize=16)
pyplot.ylabel('True Positive Rate', fontsize=16)
pyplot.title('Courbe de ROC : '+name, fontsize=16)
pyplot.legend(loc="lower right", fontsize=14)
pyplot.show()