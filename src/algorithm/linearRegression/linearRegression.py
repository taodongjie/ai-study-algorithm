import numpy as np

from src.algorithm.linearRegression.process_data import _pre_process_data_X


class OLSLinearRegression:
    """线性回归：最小二乘法"""

    def __init__(self):
        # 模型参数w(训练时初始化)
        self.w = None

    def train(self, X_train, y_train):
        """训练模型"""
        # 预处理X_train(添加x0=1)
        X_train = _pre_process_data_X(X_train)

        # 使用最小二乘法估算w
        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        """预测"""

        # 预处理X_train(添加x0=1)
        X = _pre_process_data_X(X)
        return np.matmul(X, self.w)

    @staticmethod
    def _ols(X, y):
        """最小二乘法估算w=[(XTX)-1]XTY"""
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)


class GDLinearRegression:
    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        # 训练迭代次数
        self.n_iter = n_iter

        # 学习率
        self.eta = eta

        # 误差变化阈值
        self.tol = tol

        # 模型参数w(训练时初始化)
        self.w = None

    def train(self, X_train, y_train):
        """训练"""
        # 预处理X_train(添加x0=1)
        X_train = _pre_process_data_X(X_train)

        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05

        # 执行梯度下降训练w
        self._gradient_descent(self.w, X_train, y_train)

    def predict(self, X):
        """预测"""
        X = _pre_process_data_X(X)
        return self._predict(X, self.w)

        pass

    def _loss(self, y, y_pred):
        """计算损失：mse"""
        return np.sum((y_pred - y) ** 2) / y.size

    def _gradient(self, X, y, y_pred):
        """计算梯度"""
        return np.matmul(y_pred - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        """梯度下降算法"""

        loss_old = np.inf
        # 使用梯度下降，至多迭代n_iter次，更新w
        for step_i in range(self.n_iter):
            # 预测
            y_pred = self._predict(X, w)

            # 计算损失
            loss = self._loss(y, y_pred)
            print('%4i Loss: %s' % (step_i, loss))

            # 若用户指定tol,则启用早期停止法
            if self.tol is not None:
                # 如果损失下降小于阈值，则终止迭代
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            # 计算梯度
            grad = self._gradient(X, y, y_pred)

            # 更新参数w
            w -= self.eta * grad

    @staticmethod
    def _predict(X, w):
        """预测内部接口，实现函数h(x)."""
        return np.matmul(X, w)
        pass
