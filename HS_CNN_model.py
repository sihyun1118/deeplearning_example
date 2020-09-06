import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D
from keras import models
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=50)
L_enc = LabelEncoder()
O_enc = OneHotEncoder()
df = pd.read_excel('C:/Users/sihyun/Desktop/해성옵틱스 PJ/test/final_test.xlsx')

# Model 1
x = df.iloc[:, 0:-7]
y = df.iloc[:, -7]

x = scale(x)
x_S = x
x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A1 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result1 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)

for i in range(1, len(result_A)+1):
    ind = np.sort(result_A[i - 1])[::-1]
    a = np.where(result_A[i - 1] == ind[0])
    A = a[0] * 45
    A = A[0].astype(str)
    globals()['List{}'.format(i)] = []
    globals()['List{}'.format(i)].append(A)

# Model 2


x_A1 = df.iloc[:, -7]
y = df.iloc[:, -6]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_s = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A2 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result2 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)
list_append(result_A)

# Model 3

x_A1 = df.iloc[:, -6]
y = df.iloc[:, -5]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_S = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A3 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result3 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)
list_append(result_A)

# Model 4

x_A1 = df.iloc[:, -5]
y = df.iloc[:, -4]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_S = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A4 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result4 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)
list_append(result_A)

# Model 5
x_A1 = df.iloc[:, -4]
y = df.iloc[:, -3]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_S = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A5 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result5 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)
list_append(result_A)

# Model 6
x_A1 = df.iloc[:, -3]
y = df.iloc[:, -2]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_S = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A6 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result6 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
# acc_plot(x_train, y_train, results)
acc(x_train, y_train)
list_append(result_A)

# Model 7
x_A1 = df.iloc[:, -2]
y = df.iloc[:, -1]

x_A1_E = L_enc.fit_transform(x_A1)
x_A1_E = x_A1_E.reshape(-1, 1)
x_A1 = O_enc.fit_transform(x_A1_E).toarray()
x_A1 = pd.DataFrame(x_A1)
x = pd.concat([x_S, x_A1], axis=1)
x_S = x

x_train, x_test, y_train, y_test, n_col = split(x, y)
test_A7 = y_test
x_train, x_test = expand(x_train, x_test)
y_train, y_test = encoding(y_train, y_test)
model, result7 = CNN_model(x_train, y_train)
result_A = model.predict(x_test)
acc(x_train, y_train)
# acc_plot(x_train, y_train, results)
list_append(result_A)


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



def scale(x):
    scale = StandardScaler()
    x = scale.fit_transform(x)
    x = pd.DataFrame(x)
    return x
def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    n_col = len(x_train.columns)
    return x_train, x_test, y_train, y_test, n_col
def expand(x_train, x_test):
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, x_test
def encoding(y_train, y_test):
    E_y_train = L_enc.fit_transform(y_train)
    E_y_test = L_enc.fit_transform(y_test)

    E_y_train = E_y_train.reshape(-1, 1)
    y_train = O_enc.fit_transform(E_y_train).toarray()
    E_y_test = E_y_test.reshape(-1, 1)
    y_test = O_enc.fit_transform(E_y_test).toarray()
    return y_train, y_test

def CNN_model(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(n_col, 1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    results = model.fit(x_train, y_train, batch_size=512, epochs=200, verbose=0, callbacks=[es], validation_split=0.2)
    return model, results
def acc(x_train, y_train):
    acc = model.evaluate(x_train, y_train)
    print("Loss: ", round(acc[0], 3), "Accuracy :", round(acc[1] * 100, 2), "%")
def acc_plot(x_train,y_train, results):
    acc = model.evaluate(x_train, y_train)
    print("Loss: ", round(acc[0], 3), "Accuracy :", round(acc[1] * 100, 2), "%")
    plt.figure(figsize=[15, 5])



    # plt.savefig('CRNN_56.png', dpi = 1000)
def list_append(result_A):
    for i in range(1, len(result_A)+1):
        ind = np.sort(result_A[i - 1])[::-1]
        a = np.where(result_A[i - 1] == ind[0])
        A = a[0] * 45
        A = A[0].astype(str)
        globals()['List{}'.format(i)].append(A)


######################################################################################
    for i in range(1, 8):
        plt.plot(globals['result{}'.format(i)].history['accuracy'])

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
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()










# Use scikit-learn to grid search the batch size and epochs

from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, AveragePooling1D

model = Conv1D(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

########################################################################################################################

from keras.models import load_model
model.save('CNN_Model_final.h5')