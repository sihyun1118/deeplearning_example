# 정규방정식 test
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)

import matplotlib.pyplot as plt
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

# sklearn 선형 회귀
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

# beta(theta)값 찾기
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
np.linalg.pinv(X_b).dot(y)

# 경사하강법의 스템
# batch gradient descent
# 학습률
eta = 0.1
n_iteration = 1000
m = 100

theta = np.random.randn(2,1)
for iteration in range(n_iteration):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
theta

# randomly gradient descent
n_epochs = 50
# 학습 스케줄 hyperparameter
t0, t1 = 5, 50
m = 100
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2, 1)
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
theta

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y)

# 다항 회귀
# 비선형 데이터 생성
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

# 학습곡선 살펴보는 그래프 만들기
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train-set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val-set")
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)

# 10차 다항 회귀 모델의 학습 곡선
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([("poly_features", PolynomialFeatures(degree=10, include_bias=False)), ("lin_reg", LinearRegression()), ])
plot_learning_curves(polynomial_regression, X, y)

# 너비 기반 Iris-Virsinica 종을 감지하는 분류기
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
# 꽃잎의 너비
X = iris["data"][:, 3:]

# Virsinica 종
y = (iris["target"] == 2).astype(np.int)
# 로지스틱 회귀 모형 훈련
from sklearn.linear_model import LogisticRegression
log_reg = LinearRegression()
log_reg.fit(X, y)

# 꽃잎의 너비가 0~3cm인 꽃에 대해 모델의 추정 확률 계산
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict(X_new)
plt.plot(X_new, y_proba[:, 1], "g--", label="Iris Virsinica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris Virsinica")
