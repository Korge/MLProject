import pandas as pd
import numpy
from sklearn.model_selection import train_test_split


#Ce script permet de split notre base d'apprentisage en base de train et de validation. Passage encodage des quali en quanti ( 1 modalité = 1 colonne)

data = pd.read_csv("adult.data",names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"],sep=r'\s*,\s*', engine='python', na_values="?")
data = data.dropna(axis=0) #Après cette manip, on a 30 162 enregistrements
data = pd.get_dummies(data)

# x_train, x_valid, y_train, y_valid = train_test_split(data.iloc[:,0:104].values,data.iloc[:,104:106].values,train_size=0.85,test_size=0.15)
#
# numpy.savetxt("x_train", x_train, delimiter=",",fmt="%i")
# numpy.savetxt("x_valid", x_valid, delimiter=",",fmt="%i")
# numpy.savetxt("y_train", y_train, delimiter=",",fmt="%i")
# numpy.savetxt("y_valid", y_valid, delimiter=",",fmt="%i")
