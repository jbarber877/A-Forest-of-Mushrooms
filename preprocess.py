import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split

data= pd.read_csv("mushrooms.csv") 

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