# Decision Tree Classifier
import pandas
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# Load the Iris Datasets 1
dataset1 = datasets.load_iris()
print("<<<<< First Dataset - without pandas >>>>>")
print(dataset1.data)
print(dataset1.target)
print(" ")


# Load the Iris Datasets 2
names = ['sepal-lenght', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset2 = pandas.read_csv("iris_data.csv", names=names)
print("<<<<< Second Dataset - with pandas >>>>>")
print(dataset2.shape)
print(dataset2.head(20))
print(dataset2.describe())
print(" ")


# Fit a CART model to the data
# model = classificador
model = DecisionTreeClassifier()


# ajusta o modelo sobre os dados
# dataset.data = dados de treinamento / dataset.target = alvo de treinamento
model.fit(dataset1.data, dataset1.target)
print(model)
print(" ")


# Make Predictions
expected = dataset1.target
predicted = model.predict(dataset1.data)


# Summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

