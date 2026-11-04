import numpy as np

def divide_on_feature(X, feature_i, threshold):
	# Your code here
	ups, downs = [], []
    for i in range(len(X)):
        if X[i, feature_i] >= threshold:
            ups.append(i)
        else:
            downs.append(i)
    return [X[ups], X[downs]]