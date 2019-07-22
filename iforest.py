import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        numX, numQ = X.shape
        self.trees = []
        height_limit = np.log2(self.sample_size)
        for i in range(self.n_trees):
            idx = np.random.randint(numX, size=self.sample_size)
            X_sample = X[idx, :]
            self.trees.append(IsolationTree(height_limit).fit(X_sample, improved=improved))
        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        numX, numQ = X.shape
        avg_path_length_list = []
        for i in range(numX):
            obs = X[i, :]
            path_length_list = []
            for t in self.trees:
                current_path_length = 0
                path_length_list.append(
                    self.onePathLength(obs, t, current_path_length))
            avg_path_length_list.append([np.mean(path_length_list)])
        return np.array(avg_path_length_list)

    def onePathLength(self, obs, tree, current_path_length):
        if type(tree) == exNode:
            return current_path_length + self.c(tree.size)
        else:
            a = tree.splitAtt
            current_path_length += 1
            if obs[a] < tree.splitValue:
                return self.onePathLength(obs, tree.left, current_path_length)
            else:
                return self.onePathLength(obs, tree.right, current_path_length)

    def c(self, num):
        if num > 2:
            return 2 * (np.log(num - 1) + 0.5772156649) - 2 * (num - 1) / num
        elif num == 2:
            return 1
        else:
            return 0

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        c_psi = self.c(self.sample_size)
        score_array = 2 ** (-(self.path_length(X) / c_psi))
        return score_array

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return np.array([1 if score >= threshold else 0 for score in scores])

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)

class inNode:
    def __init__(self, splitAtt, splitValue, left=None, right=None):
        self.left = left
        self.right = right
        self.splitAtt = splitAtt
        self.splitValue = splitValue

class exNode:
    def __init__(self, size):
        self.size = size

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        _, numQ = X.shape
        current_height = 0
        root = self.iTree(X, current_height, improved = improved)
        current_height = 1
        self.grow(root, current_height, improved = improved)
        n_nodes = self.countNode(root)
        self.root = root
        self.root.n_nodes = n_nodes
        return self.root

    def iTree(self, X: np.ndarray, current_height, improved = False):
        if current_height >= self.height_limit or X.shape[0] <= 1:
            return exNode(X.shape[0])
        else:
            numX, numQ = X.shape
            splitAtt = np.random.randint(numQ, size=1)[0]
            attMax = max(X[:, splitAtt])
            attMin = min(X[:, splitAtt])
            if attMax == attMin:
                return exNode(X.shape[0])
            if improved == True:
                attMean = np.mean(X[:, splitAtt])
                if attMax - attMean >= attMean - attMin:
                    splitValue = np.random.uniform(attMean, attMax, 1)[0]
                else:
                    splitValue = np.random.uniform(attMin, attMean, 1)[0]
            else:
                splitValue = np.random.uniform(attMin, attMax, 1)[0]
            left_idx = set([i for i in range(numX) if X[i, splitAtt] < splitValue])
            right_idx = set([i for i in range(numX)]) - left_idx
            left = X[list(left_idx), :]
            right = X[list(right_idx), :]
            return inNode(splitAtt, splitValue, left, right)

    def grow(self, node, current_height, improved = False):
        if type(node) == inNode:
            node.left = self.iTree(node.left, current_height, improved = improved)
            node.right = self.iTree(node.right, current_height, improved = improved)
            current_height += 1
            self.grow(node.left, current_height, improved = improved)
            self.grow(node.right, current_height, improved = improved)

    def countNode(self, root):
        if type(root) == inNode:
            return 1 + self.countNode(root.left) + self.countNode(root.right)
        if type(root) == exNode:
            return 1

def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    for i in range(100):
        y_pred = np.array([1 if score >= threshold else 0 for score in scores])
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR
        threshold -= 0.01

