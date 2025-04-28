import numpy as np
import pandas as pd

# Calculate entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Calculate information gain
def information_gain(X_column, y):
    parent_entropy = entropy(y)
    values, counts = np.unique(X_column, return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(y[X_column == values[i]]) 
        for i in range(len(values))
    ])
    return parent_entropy - weighted_entropy

# Simple Decision Tree Classifier
class DecisionTreeClassifierManual:
    def __init__(self, depth=0, max_depth=5):
        self.depth = depth
        self.max_depth = max_depth
        
    def fit(self, X, y):
        if len(np.unique(y)) == 1 or self.depth >= self.max_depth:
            self.label = np.bincount(y).argmax()
            return
        
        gains = [information_gain(X[:, i], y) for i in range(X.shape[1])]
        self.feature = np.argmax(gains)
        
        feature_values = np.unique(X[:, self.feature])
        self.children = {}
        
        for value in feature_values:
            idx = X[:, self.feature] == value
            child = DecisionTreeClassifierManual(depth=self.depth + 1, max_depth=self.max_depth)
            child.fit(X[idx], y[idx])
            self.children[value] = child
            
    def predict(self, X):
        if hasattr(self, 'label'):
            return self.label
        child = self.children.get(X[self.feature])
        if child:
            return child.predict(X)
        else:
            # If unseen value, return majority
            return np.random.choice(list(self.children.values())).label

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

# Example usage
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

# Only take a small subset (e.g., class 0 and 1) for simplicity
idx = y < 2
X, y = X[idx], y[idx]

tree = DecisionTreeClassifierManual(max_depth=3)
tree.fit(X, y)

y_pred = tree.predict_batch(X)
print("Manual Decision Tree Accuracy:", np.mean(y_pred == y))
