import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, Activation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
CNN = True
 
def output_gray_image(df, i):
    img = df.drop(["label"], axis = 1).iloc[i].values
    img = img.reshape((28,28))
    plt.imshow(img, cmap='gray')
 
def output_binary_image(df, i):
    add_lambda = lambda x: 1 if int(x) > 100 else 0
    j = 0
    for m in df.drop(["label"], axis=1).iloc[i]:
        print(add_lambda(m), end="")
        j = j + 1
        if (j % 28 == 0):
           print("\n", end = "")
 
# 学習ネットワーク構築（Functional API）
def build_model(input_dim):
    inputs = Input(shape = (input_dim,))
    layer  = Dense(64, activation = 'relu')(inputs)
    dense  = Dense(10, activation = 'softmax')(layer)
    model  = Model(inputs = inputs, outputs = dense)
    return model
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
 
# 畳込みニューラルネットワーク構築（Sequential Version）
def build_model_cnn2(input_dim):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), 
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model
 
# 畳込みニューラルネットワーク構築（Functional API）
def build_model_cnn(input_dim):
    inputs = Input(shape=(28, 28, 1))
    layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    layer2 = Conv2D(64, (3, 3), activation='relu')(layer1)
    layer3 = MaxPooling2D(pool_size=(2, 2))(layer2)
    layer4 = Dropout(0.25)(layer3)
    layer5 = Flatten()(layer4)
    layer6 = Dense(128, activation='relu')(layer5)
    layer7 = Dropout(0.5)(layer6)
    output = Dense(10, activation='softmax')(layer7)
    model  = Model(inputs=inputs, outputs=output)
    return model
 
# 学習用データ x_data、検証用結果 y_data の割り当て
def build_train_test_data(df_train):
       # 答えの削除
       y_train = df_train["label"]
       X_train = df_train.drop(labels = ["label"],axis = 1)
       # 0~255を0~1に正規化
       X_train = X_train / 255.0
       if (CNN):
           # 1×784→28×28に変換(1次元→2次元に変換)
           # TensorFlowでは4次元配列（サンプル数, 画像の行数, 画像の列数, チャネル数）
           X_train = X_train.values.reshape(-1, 28, 28, 1)
       #ラベルをone hot vectorsに (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
       y_train = to_categorical(y_train, num_classes = 10) 
       # データを訓練用（学習用）とテスト用に分割
       X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8)# 1×784→28×28に変換(1次元→2次元に変換)
       # 作成した行列出力
       print("x_train:{}".format(X_train.shape))
       print("y_train:{}".format(y_train.shape))
       print("x_test:{}".format(X_test.shape))
       print("y_test:{}".format(y_test.shape))
       return X_train, X_test, y_train, y_test
 
def main():
       # CSV読み込み
       df_train = pd.read_csv("train.csv")
       # 学習ネットワーク
       start = time.time()
       X_train, X_test, y_train, y_test = build_train_test_data(df_train)
       if (CNN):
           model = build_model_cnn(X_train.shape[1])
           epochs = 12
           batch_size = 128
       else:
           model = build_model(X_train.shape[1])
           epochs = 120
           batch_size = 16
       model.summary()
       plot_model(model, to_file = 'model3.png', show_shapes = True)
       # 最適化手法はAdamを使う
       model.compile(loss = "categorical_crossentropy", optimizer = Adam(), metrics = ["accuracy"])
       # 訓練データを学習
       model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
       # 結果出力
       [loss, accuracy] = model.evaluate(X_test, y_test, verbose = 0)
       print("loss:{0} -- accuracy:{1}".format(loss, accuracy))
       print(str(time.time() - start))
 
       # テスト値を読み込み
       df_test = pd.read_csv("test.csv")
       # 0~255を0~1に正規化
       X_test = df_test / 255.0
       if (CNN):
           # 1×784→28×28に変換(1次元→2次元に変換)
           X_test = X_test.values.reshape(-1, 28, 28, 1)
       predictions = model.predict(X_test)
       # one-hotベクトルで結果が返るので、数値に変換する
       df_out = [np.argmax(v, axis = None, out = None) for v in predictions]
       # 整形
       df_out = pd.Series(df_out, name = "Label")
       submission = pd.concat([pd.Series(range(1, df_test.shape[0] + 1), name = "ImageId"),df_out],axis = 1)
       # CSVに出力する
       submission.to_csv("submission.csv",index=False)
 
main()
