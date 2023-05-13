import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_split=2, max_depth=100, num_features=None):
        self.min_split=min_split
        self.max_depth=max_depth
        self.num_features=num_features
        self.root=None

    def fit(self, X, y):
        self.num_features = X.shape[1] if not self.num_features else min(X.shape[1],self.num_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_feats = X.shape
        num_labels = len(np.unique(y))

        # check the stopping criteria
        if (depth>=self.max_depth or num_samples<self.min_split or num_labels==1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        indexs = np.random.choice(num_feats, self.num_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, indexs)

        # create child nodes
        left_index, right_index = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_index, :], y[left_index], depth+1)
        right = self._grow_tree(X[right_index, :], y[right_index], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, indexs):
        curr_gain = -1
        split_idx, split_threshold = None, None

        for index in indexs:
            X_column = X[:, index]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > curr_gain:
                    curr_gain = gain
                    split_idx = index
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_index, right_index = self._split(X_column, threshold)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        weight_left = len(left_index)/len(y)
        weight_right = len(right_index)/len(y)
        entropy_left = self._entropy(y[left_index])
        entropy_right = self._entropy(y[right_index])
        child_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _split(self, X_column, split_thresh):
        left_index = np.argwhere(X_column <= split_thresh).flatten()
        right_index = np.argwhere(X_column > split_thresh).flatten()
        return left_index, right_index
    
    def _entropy(self, y):
        ps = np.bincount(y) / len(y)
        return (-1 * np.sum([p * np.log(p) for p in ps if p>0]))

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])

    def traverse(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse(x, node.left)
        return self.traverse(x, node.right)