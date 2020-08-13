import os

os.chdir('C:/Users/sihyun/PycharmProjects/HandsOn')
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape
y.shape

import matplotlib as mpl
import matplotlib.pyplot as plt
# 샘플의 특성 벡터 추출
some_digit = X[0]
# 28x28 배열로 크기 바꾸기
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
# ?????
plt.axis("off")
# ?????
plt.show()
# 5 / 문자형임
y[0]
# 정수로 변환
y = y.astype(np.uint8)
# dataset 나누기
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# # dataset 섞기 (MNIST는 이미 섞여 있으므로 주석처리)
# import numpy as np
# shuffle_index = np.random.permutation(60000)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# 5-감지기(이진 분류기) -> '5' or 'not 5' / true or false
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# SGD(확률적 경사 하강법 분류기) -> 큰 dataset을 효율적으로 처리 / 온라인 학습에 좋음
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

# sklearn에서 제공하는 cross_val_socre()와 같은 기능 but 교차 검정 과정에서 많은 제어가 필요할 때 사용
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct/len(y_pred))

# sklearn cross_val_score()
# 'not 5' 클래스로 분류하는 더미 분류기
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self,X,y=None):
        return self
    def predict(self,X):
        return np.zeros((len(X),1), dtype = bool)
# 모델 정확도 추측
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5,cv=3, scoring="accuracy")

# 오차행렬을 만들기 위한 예측값 만들기
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# 오차 행렬 만들기
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
# 완벽한 분류기인 경우
y_train_perfect_prediction = y_train_5
confusion_matrix(y_train_5, y_train_perfect_prediction)

# 분류기의 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
# 정밀도와 재현율을 사용한 F1-score
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

# 각 샘플의 점수를 얻어 원하는 임곗값을 정해 예측
y_score = sgd_clf.decision_function([some_digit])
y_score
threshold = 0
y_some_digit_pred = (y_score> threshold)
threshold = 8000

# 적절한 임곗값을 정하기
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# y_scores로 가능한 모든 임곗값의 정밀도와 재현율 계산
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
precisions.shape
recalls.shape
thresholds.shape
# matplotlib.pyplot을 이용해서 임곗값의 함수로 정밀도와 재현율 그리기
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],"b--", label="정밀도")
    plt.plot(thresholds, recalls[:-1], "g--", label="재현율")
    [...]

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

threshold_90_precision = thresholds[np.argmax(precisions>=0.90)]
y_train_pred_90 = (y_scores >=threshold_90_precision)
precision_score(y_train_5,y_train_pred_90)
recall_score(y_train_5,y_train_pred_90)

# ROC 곡선
# TPR과 FPR 계산
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# TPR에 대한 FPR 곡선
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    [...]
plot_roc_curve(fpr,tpr)
plt.show()
# ROC의 AUC 계산
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

# RF Classifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# 양성 클래스에 대한 확률을 점수로 사용
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
# ROC 곡선 그리기
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "RF")
plt.legend(loc="lower right")
plt.show()
# AUC 계산
roc_auc_score(y_train_5, y_scores_forest)

# 다중분류
# SVC
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores
np.argmax(some_digit_scores)
svm_clf.classes_
svm_clf.classes_[5]

# SVC기반 OvR전략
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict(([some_digit]))
len(ovr_clf.estimators_)

# SGDClassfier 훈련 -> 다중분류기이므로 OvR이나 OvO를 적용할 필요 X
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])

#분류기 평가
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# 입력의 scale을 조정하여 정확도 향상 시키기
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# Error 분석
# 오차행력 만들기
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
# 오차행렬 시각화
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# 그래프 에러 부분에 초점을 맞추기
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
# 주 대각선만 0으로 채워서 그래프 그리기
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b = 3,5
X_aa = X_train[(y_train ==cl_a) & (y_train_pred ==cl_a)]
X_ab = X_train[(y_train ==cl_a) & (y_train_pred ==cl_b)]
X_ba = X_train[(y_train ==cl_b) & (y_train_pred ==cl_a)]
X_bb = X_train[(y_train ==cl_b) & (y_train_pred ==cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], image_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], image_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], image_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], image_per_row=5)
plt.show()

# 다중 레이블 분류
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)


# noise 추가
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)