import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

x_train = pd.read_csv('x_train')
x_valid = pd.read_csv('x_valid')
y_train = pd.read_csv('y_train')
y_valid = pd.read_csv('y_valid')

# parameters = {
#     'max_depth':(2,3,6,10,13),
#     'min_samples_split':(2,5,10,15,25,40,60)
# }
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