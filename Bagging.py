import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
import pickle

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

# parameters = {
#     'n_estimators':(11,19,29,39),
#     'oob_score':(False,True)
# }
#
# myTree = pickle.load(open("BestTree","rb")) #On récupère le meilleur arbre que nous avions trouvé
# myBag = BaggingClassifier(myTree)
# Grid = GridSearchCV(myBag,parameters,cv=3,verbose=1)
# print("GridSearch launched")
# Grid.fit(x_train,y_train.iloc[:,1].values)
# myBag = Grid.best_estimator_
# pickle.dump(myBag, open("BestBag","wb"))

myBag = pickle.load(open("BestSVM","rb"))
print("Best Bagging selected is : ")
print(myBag.get_params)
print("Accuracy on training set : "  + str(myBag.score(x_train,y_train.iloc[:,1].values)))
print("Accuracy on valid set : "  + str(myBag.score(x_valid,y_valid.iloc[:,1].values)))