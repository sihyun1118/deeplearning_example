# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import sklearn
# from sklearn.linear_model import LinearRegression
#
# # 데이터 적재
# OECD_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
# gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")
#
# # 함수 정의
# def prepare_country_stats(oecd_bli, gdp_per_capita):
#     oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
#     oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
#     gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
#     gdp_per_capita.set_index("Country", inplace=True)
#     full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
#                                   left_index=True, right_index=True)
#     full_country_stats.sort_values(by="GDP per capita", inplace=True)
#     remove_indices = [0, 1, 6, 8, 33, 34, 35]
#     keep_indices = list(set(range(36)) - set(remove_indices))
#     return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
#
# # 데이터 준비
# country_stats = prepare_country_stats(OECD_bli, gdp_per_capita)
# X = np.c_[country_stats["GDP per capita"]]
# Y = np.c_[country_stats["Life satisfaction"]]
#
# # 데이터 시각화
# country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
# plt.show()
#
# # 선형 모델 선택
# model = sklearn.linear_model.LinearRegression()
#
# #모델 훈련
# model.fit(X, Y)
#
# # 키프로스에 대한 예측
# X_new = [[22587]]
# print(model.predict(X_new))
#
# # 코드 예시 K-Nearest Neighbors 모델링
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)


# chap2
import matplotlib
import scipy
import sklearn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
os.getcwd()
# CSV파일 불러오기
# data info(type, count..)
housing = pd.read_csv('housing.csv')
housing.info()
# data summary
housing.describe()
housing.hist(bins=50,figsize=(20,15))

# 데이터 세트 만들기
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set),len(test_set))

# 프로그램을 다시 실행해도 같은 dataset으로 사용하기
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# index 열 추가
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# ID 만들기
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# 소득의 카테고리 수를 제한하기 위해
housing["income_cat"] = np.ceil(housing["medianIncome"]/1.5)
# 이산적인 카테고리를 만들기 위해
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace = True)

# 소득 카테고리를 기반으로 계층 샘플링하기
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts()/len(housing)

# income_cat 특성을 삭제해서 데이터를 원상태하기
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis = 1, inplace = True)
# training set를 손삿시키지 않기 위해 복사본 만들기
housing = strat_train_set.copy()

# 산점도 만들기
housing.plot(kind = "scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label = "population", figsize=(10, 7),
             c="medianHouseValue", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()

# 상관관계 조사
corr_matrix = housing.corr()
corr_matrix["medianHouseValue"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["medianHouseValue","medianIncome","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))

housing.plot(kind="scatter", x="medianIncome", y="medianHouseValue", alpha=0.1)

#특성 조합으로 실험
#가구당 방 개수
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# 방당 침대 개수
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# 가구당 인구수
housing["population_per_households"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["medianHouseValue"].sort_values(ascending=False)

housing = strat_train_set.drop("medianHouseValue", axis = 1)
housing_labels = strat_train_set["medianHouseValue"].copy()

# NA값 처리하기
# 1. 해당 구역 제거
housing.dropna(subset=["total_bedrooms"])
# 2. 전체 특성 삭제
housing.drop("total_bedrooms", axis =1)
# 3. 대표값으로 대체
medain = housing["total_bedrooms"].medain()
housing["total_bedrooms"].fillna(median, inplace = True)
# 4. sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("")
SimpleImputer.fit(housing)