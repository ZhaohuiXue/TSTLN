import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

def calculate_metrics(y_true, y_pred):

    mat = confusion_matrix(y_true, y_pred)
    Kappa = cohen_kappa_score(y_true, y_pred)
    OA = accuracy_score(y_true, y_pred)*100
    PA0 = mat[0, 0] / sum(mat[0, :])*100
    PA1 = mat[1, 1] / sum(mat[1, :])*100
    PA2 = mat[2, 2] / sum(mat[2, :])*100
    CA0 = mat[0, 0] / sum(mat[:, 0])*100
    CA1 = mat[1, 1] / sum(mat[:, 1])*100
    CA2 = mat[2, 2] / sum(mat[:, 2])*100

    acc = [OA, Kappa, PA0, PA1, PA2, CA0, CA1, CA2]

    return acc