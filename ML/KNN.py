from numpy import *
import operator as op


def create_data():
    data = array([[1, 1], [1, 1.1], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return data, label


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
