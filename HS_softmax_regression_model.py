import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.layers import Dense, Dropout, Embedding
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=10)
L_enc = LabelEncoder()
O_enc = OneHotEncoder()

df = pd.read_excel('C:/Users/sihyun/Desktop/해성옵틱스 PJ/test/final_test.xlsx')

# M1
x = df.iloc[:, 0:-7]
n_col = len(x.columns)
y = df.iloc[:, -7]

scale = StandardScaler()
x = scale.fit_transform(x)
x = pd.DataFrame(x)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A1 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result1 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)] = []
    globals()['List{}'.format(i)].append(A)


# M2
x_A1 = df.iloc[:, -7]
y = df.iloc[:, -6]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A2 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result2 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)
x_train.shape
#M3
x_A1 = df.iloc[:, -6]
y = df.iloc[:, -5]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A3 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result3 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)
x_train.shape
#M4
x_A1 = df.iloc[:, -5]
y = df.iloc[:, -4]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A4 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result4 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)

# M5
x_A1 = df.iloc[:, -4]
y = df.iloc[:, -3]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A5 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result5 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)
x_train.shape
#M6
x_A1 = df.iloc[:, -3]
y = df.iloc[:, -2]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A6 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result6 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)
x_train.shape
# M7
x_A1 = df.iloc[:, -2]
y = df.iloc[:, -1]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)

x = pd.concat([x_S, x_A1], axis=1)
n_col = len(x.columns)
x_S = x
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_A7 = y_test
E_y_train = L_enc.fit_transform(y_train)
E_y_test = L_enc.fit_transform(y_test)
E_y_train = E_y_train.reshape(-1, 1)
y_train = O_enc.fit_transform(E_y_train).toarray()
E_y_test = E_y_test.reshape(-1, 1)
y_test = O_enc.fit_transform(E_y_test).toarray()
model = Sequential()
model.add(Dense(8, input_dim=n_col, activation='softmax'))
sgd = optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result7 = model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2,callbacks=[es])
result_A = model.predict(x_test)
for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)].append(A)
x_train.shape


df_A = pd.DataFrame(List1)
df_A = df_A.T
for i in range(2, 200):
    L = pd.DataFrame(globals()['List{}'.format(i)])
    A_list = L.T
    df_A = pd.concat([df_A, A_list])

test_A = pd.DataFrame(test_A1)
for i in range(2, 8):
    L = pd.DataFrame(globals()['test_A{}'.format(i)])
    test_A = pd.concat([test_A, L], axis=1)
test_A = test_A.astype(str)


test_A['C'] = test_A['A1'].astype(str)+'-'+test_A['A2'].astype(str)+'-'+test_A['A3'].astype(str)+'-'+test_A['A4'].astype(str)+'-'+test_A['A5'].astype(str)+'-'+test_A['A6'].astype(str)+'-'+test_A['A7'].astype(str)
df_A['C'] = df_A[0].astype(str)+'-'+df_A[1].astype(str)+'-'+df_A[2].astype(str)+'-'+df_A[3].astype(str)+'-'+df_A[4].astype(str)+'-'+df_A[5].astype(str)+'-'+df_A[6].astype(str)
count = 0
for i in range(result_A.shape[0]):
    pre = df_A['C'].iloc[i]
    ori = test_A['C'].iloc[i]
    if (pre == ori):
        count += 1

combi_acc = count / (result_A.shape[0])
print(combi_acc*100)
x_train.shape

from keras.models import load_model
model.save('Softmax_Regression.h5')

plt.plot(result1.history['accuracy'])
plt.plot(result2.history['accuracy'])
plt.plot(result3.history['accuracy'])
plt.plot(result4.history['accuracy'])
plt.plot(result5.history['accuracy'])
plt.plot(result6.history['accuracy'])
plt.plot(result7.history['accuracy'])
plt.plot(result1.history['val_accuracy'])
plt.plot(result2.history['val_accuracy'])
plt.plot(result3.history['val_accuracy'])
plt.plot(result4.history['val_accuracy'])
plt.plot(result5.history['val_accuracy'])
plt.plot(result6.history['val_accuracy'])
plt.plot(result7.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(result1.history['loss'])
plt.plot(result2.history['loss'])
plt.plot(result3.history['loss'])
plt.plot(result4.history['loss'])
plt.plot(result5.history['loss'])
plt.plot(result6.history['loss'])
plt.plot(result7.history['loss'])
plt.plot(result1.history['val_loss'])
plt.plot(result2.history['val_loss'])
plt.plot(result3.history['val_loss'])
plt.plot(result4.history['val_loss'])
plt.plot(result5.history['val_loss'])
plt.plot(result6.history['val_loss'])
plt.plot(result7.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()