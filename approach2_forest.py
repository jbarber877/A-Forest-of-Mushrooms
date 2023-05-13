import numpy as np 
import pandas as pd

from tree import DecisionTree
from forest import RandomForest
from preprocess import X_train, X_test, y_train, y_test, accuracy

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import classification_report

print("Custom random forest classifier")
clf = RandomForest(num_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)

print(classification_report(y_test,predictions))

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    
cm_display.plot()
plt.show()

print("Scikit learn's random forest classifier")
clf = RandomForestClassifier(n_estimators = 20,
                             criterion = 'entropy',
                             max_depth=10,
                             min_samples_split=2,
                             random_state=1234)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)

print(classification_report(y_test,predictions))

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    
cm_display.plot()
plt.show()