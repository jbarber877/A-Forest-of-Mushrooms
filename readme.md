
# A Forest Of Mushrooms 

Our goal was to implement a Random Forest Classifier from scratch and compare it to sklearn’s RandomForestClassifier() in terms of performance and accuracy. Both models used the same data to classify if a mushroom was poisonous or not.
We created a basic Decision Tree and Random Forest Classifier. We  also imported sklearn’s models and were able to classify the data using both methods.



## Technical Specifications 

----- \
matplotlib          3.7.0 \
numpy               1.23.5 \
pandas              1.5.3 \
session_info        1.0.0 \
sklearn             1.2.1 

----- \
IPython             8.12.0 \
jupyter_client      8.2.0 \
jupyter_core        5.3.0 \
jupyterlab          3.5.3 \
notebook            6.5.4 

Python 3.10.9 | packaged by Anaconda, Inc. | (main, Mar  1 2023, 18:18:15) [MSC v.1916 64 bit (AMD64)] \
Windows-10-10.0.19044-SP0 \

Session information updated at 2023-04-24 10:30

## Setup guide for development

```bash
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import metrics
from sklearn.metrics import classification_report

from matplotlib.colors import ListedColormap
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

```
Test data population
The data used in this project comes from the UCI Machine Learning repository available here (https://archive.ics.uci.edu/ml/datasets/mushroom).


```bash
data= pd.read_csv("mushroomsTwo.csv") 
```
The test and train data was used to compare the following classifiers: Decision Tree, Random Forest, and sklearn's models

```bash
mush = pd.get_dummies(data=data, columns=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
           'ring-type', 'spore-print-color', 'population', 'habitat'], drop_first=True)


X = mush.drop('class_p', axis=1)
y = mush['class_p']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)
```
Decision Tree Test
```bash
%%time
clf = DecisionTree(max_depth = 10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print(acc)
```
Random Forest Test
```bash
%%time
clf = RandomForest(num_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)
```
RandomForestClassifier Test
```bash
%%time
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 20,
                             criterion = 'entropy',
                             max_depth=10,
                             min_samples_split=2,
                             random_state=1234)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)
```

## Results
Results show that the random forest model achieved high accuracy at classifying mushrooms, making it a reliable tool for identifying dangerous mushrooms in the future.
Although our models did not match the accuracy or speed of sklearn, they did correctly classify mushrooms as poisonous or edible.
## Authors

Jared Barber
 
Arunabh Bhattacharya 

## Reference:
https://www.kaggle.com/datasets/uciml/mushroom-classification