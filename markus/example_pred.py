"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random
from data_processing import preprocess_data

# numpy and pandas are also permitted
import numpy as np
import pandas as pd

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
from collections import Counter
import math


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=30, min_samples_split=2, min_samples_leaf=1, n_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Validate inputs
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Feature selection
        feat_idxs = np.random.choice(n_feats, min(self.n_features, n_feats) if self.n_features else n_feats,
                                     replace=False)

        # Find best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # If no split found, return leaf node
        if best_feature is None or best_thresh is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        # Check if split results in nodes with less than min_samples_leaf
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    left_idxs, right_idxs = self._split(X_column, thr)
                    # Only consider splits that result in both children having at least min_samples_leaf
                    if len(left_idxs) >= self.min_samples_leaf and len(right_idxs) >= self.min_samples_leaf:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate weighted avg entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0  # Return default value instead of raising error
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=100, criterion="gini", max_depth=30, min_samples_split=2, min_samples_leaf=1,
                 n_features=None):
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # Validate inputs
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Input data is empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.trees = []
        n_features = X.shape[1]

        # Calculate number of features to use
        if isinstance(self.n_features, str):
            if self.n_features == 'sqrt':
                n_feature = int(math.sqrt(n_features))
            elif self.n_features == 'log2':
                n_feature = int(math.log2(n_features))
            else:
                n_feature = None
        else:
            n_feature = self.n_features

        for _ in range(self.n_trees):
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_features=n_feature
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0  # Return default value instead of raising error
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        return np.array([self._most_common_label(pred) for pred in tree_preds])



# Load the exported data
df_new = pd.read_csv('model_data.csv')

# Separate features and target
X_new = df_new.drop(columns=['Label']).values
y_new = df_new['Label'].values

# Initialize RandomForest with correct parameters
rf_model = RandomForest(
    n_trees=100, 
    criterion='entropy', 
    max_depth=30, 
    min_samples_split=10, 
    min_samples_leaf=2,
    n_features='sqrt'
)

# Fit the model (train once)
rf_model.fit(X_new, y_new)


def preprocess_data(df):
    print('help me')



def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the three choices: 'Pizza', 'Shawarma', 'Sushi'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the food items!


    vals = ['Pizza', 'Shawarma', 'Sushi']
    matcher = {vals[0]: 0, vals[1]: 1, vals[2]: 2}
    y = vals[matcher[rf_model.predict([x])[0]]]


    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)

    # df = pd.read_csv('/content/drive/MyDrive/311/final_final_fr_fr.csv')
    df = pd.read_csv(filename)
    data = preprocess_data(df)
    

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions


