from tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, num_trees=10, max_depth=10, min_split=2, num_features=None):
        self.num_trees = num_trees
        self.max_depth=max_depth
        self.min_split=min_split
        self.num_features=num_features
        self.forest = []

    def fit(self, X, y):
        self.forest = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_split=self.min_split,
                            num_features=self.num_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)

    def _bootstrap_samples(self, X, y):
        num_samples = X.shape[0]
        index = np.random.choice(num_samples, num_samples, replace=True)
        return X[index], y[index]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.forest])
        return np.array([self._most_common_label(p) for p in (np.swapaxes(predictions, 0, 1))])