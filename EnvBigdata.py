import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

df = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/REAL_FINAL.csv",  encoding='CP949')
df.columns
df.dtype
type(df)
# 결측치 제거
del df['사고건수']
df = df.iloc[0:249, :]
df.isnull()
df['폐수발생량'].isnull().sum()
df['폐수방류량'].isnull().sum()
df['유기물질부하량(kg)일_발생'].isnull().sum()
df['유기물질부하량(kg)일_방류'].isnull().sum()

df['NaN_count'] = df.isnull().sum(1)
del df['NaN_count']


# 세종시, 계룡시, 계양구를 뺸 df / train
df_1 = df.dropna()
df_1.columns
y_1 = df_1[["폐수발생량"]]
y_2 = df_1[["폐수방류량"]]
y_3 = df_1[["유기물질부하량(kg)일_발생"]]
y_4 = df_1[["유기물질부하량(kg)일_방류"]]

X = df_1.drop(df_1.columns[[0, 1, 2, 20, 21, 22, 23]], 1)

# 세종시, 계룡시, 계양구 / test
X_test = df.drop(df_1.columns[[0, 1, 2, 20, 21, 22, 23]], 1)
Sejong = X_test.iloc[59].values
Gyeryong = X_test.iloc[157].values
Gyeyang = X_test.iloc[174].values

# Linear Regresesion

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
m1 = LR.fit(X, y_1)
m1.predict(Sejong)
X = X.apply(pd.to_numeric)
X = X.astype('float')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/REAL_FINAL.csv",  encoding='CP949')
x=x.dropna(axis=0)
x_1=x
x=x.iloc[:,3:]
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
#x_normalized=normalize(x_scaled)
x_scaled=pd.DataFrame(x_scaled)
#x_normalized=pd.DataFrame(x_normalized)

db_default = DBSCAN(eps=0.003, min_samples=20).fit(x_scaled)
labels = db_default.labels_
print(labels)

kmeans = KMeans(n_clusters=3, max_iter=200, algorithm='auto')
db_default1 = kmeans.fit(x_scaled)
labels1 = db_default1.labels_



df = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/NA_nope.csv", encoding='CP949')
import matplotlib.pyplot as plt
import seaborn as sns
# 데이터 모양 확인
df.head()
# 데이터 타입 체크
df.info()
# 데이터 NA값 체크
df.isnull().sum()
# 사고건수 col 삭제
# 나머지 NA --> Linear Regression
# boxplot 이상치 탐색
# import pandas_profiling
# from pandas_profiling import ProfileReport
#
# #profile 파일 만들기
# profile = ProfileReport(df, title="bio train data set")
#
# #html 파일로 꺼내기
# profile.to_file(output_file="bio_train_profile.html")
#
# import pandas_profiling as pp # 뒤에 pp는 해당 패키지를 pp로 호출하겠다는 의미 다른거 적어도됨
#
# data = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/NA_nope.csv")
# pp.ProfileReport(data)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/NA_nope.csv", encoding='CP949')
df.info()
df.shape
df.head()
df.columns
df.info()
# box-plot 하나씩
plt.boxplot(df['주택'])
# box-plot 여러개
df_1 = df.iloc[:, 4:len(df.columns)]
plt.boxplot(df_1)
df_1.columns

# 데이터 scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler(df_1)
df_1.dtypes
callable(df_1)
df_R_1 = df_R_1.astype(str).astype(float)
plt.boxplot(df_R_1)
# scaling 전 boxplot
df_R = pd.read_csv("C:/Users/sihyun/PycharmProjects/env_bigdata/REAL_FINAL.csv", encoding='CP949')
del df_R['사고건수']
df_R_1 = df_R.iloc[:, 4:len(df.columns)]

df_R_1 = df_R_1.astype(str).astype(float)
plt.boxplot(df_R_1["아파트"])
plt.boxplot(df_R_1["의원"])
plt.show()

# 'key', 'region', 'city', '가구', '남자', '여자', '주택', '총인구',
#        '병원', '보건소', '종합병원', '약국', '요양병원', '의원', '치과', '한의원', '가로등', '교육용',
#        '산업용', '주택용', '폐수발생량', '폐수방류량', '유기물질부하량.kg.일_발생', '유기물질부하량.kg.일_방류',
#        '자살율', '주점', '과일가게', '교습소', '학원', '음식점', '문화시설.pc..노래방..당구장..영화관.',
#        '동물병원', '목욕탕', '슈퍼마켓.편의점', '미용실', '세탁소', '스포츠교육기관', '옷가게', '정육점', '제과점',
#        '채소가게', '철물점', '커피음료점', '생활폐기물', '사업장폐기물', '아파트', '오피스텔', '단독주택',
#        '연립주택'
from pandas_profiling import ProfileReport
df_1 = df.iloc[:, 4:len(df.columns)]
profile = ProfileReport(df_1, title="Pandas 프로파일 링 보고서")
profile = ProfileReport(df_1, title='Pandas Profiling Report', explorative=True)