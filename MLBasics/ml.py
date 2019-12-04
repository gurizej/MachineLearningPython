#This is import the iris data from sklearn
from sklearn.datasets import load_iris

#Putting the data within a variable
iris = load_iris()
#Printing out their names only
print(list(iris.target_names))


####################################################################################################
from sklearn import tree

#Getting the DecisionTreeClassifier model, and putting it into a variable
classifier = tree.DecisionTreeClassifier()
#Fitting the iris data into the model
classifier = classifier.fit(iris.data, iris.target)
#Seeing under which of the three flowers the following would go into.  It will predict as to which flower it should be
print(classifier.predict([[5.1, 3.5, 1.4, 1.5]]))
