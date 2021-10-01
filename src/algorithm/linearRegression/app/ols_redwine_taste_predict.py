import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from src.algorithm.linearRegression.linearRegression import OLSLinearRegression

if __name__ == '__main__':
    # 1.数据处理
    data = np.genfromtxt('../data/winequality-red.csv', delimiter=';', skip_header=True)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    # 2.特征标准化处理
    ss = StandardScaler()
    ss.fit(X_train)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)


    # 2.模型训练
    ols_lr = OLSLinearRegression()
    ols_lr.train(X_train_std, Y_train)

    y_pred = ols_lr.predict(X_test_std)

    # 3.模型效果评估
    mse = mean_squared_error(Y_test, y_pred)
    print(mse)

    y_train_pred = ols_lr.predict(X_train_std)
    mse_train = mean_squared_error(Y_train, y_train_pred)
    print(mse_train)

