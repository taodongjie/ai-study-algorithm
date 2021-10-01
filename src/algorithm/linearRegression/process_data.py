import numpy as np


def _pre_process_data_X(X):
    """数据预处理：扩展X,添加x0并设置为1"""
    m, n = X.shape
    X_ = np.empty((m, n + 1), dtype=float, order='C')
    X_[:, 0] = 1
    X_[:, 1:] = X
    return X_
