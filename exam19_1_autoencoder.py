import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

input_img = Input(shape = (784,)) # 인풋레이어에 이미지 넣음
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation= 'sigmoid' )(encoded) # 0~1값 사이라서 시그모이드 사용함
decoded = Dense(128, activation= 'sigmoid' )(decoded)
decoded = Dense(784, activation= 'sigmoid' )(decoded)
autoencoder = Model(input_img, decoded)

# 여기까지 이미지 784개를 32개로 줄이고 decoded를 이용해 다시 784로 만들어줌
autoencoder.summary() # 50992값이 나옴


encoder = Model(input_img, encoded)
encoder.summary() # 위에 autoencoder의 인코더 부분



autoencoder.compile(optimizer='adam',loss = 'binary_crossentropy')

(x_train,_), (x_test,_) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255

flatted_x_train = x_train.reshape(-1, 28*28) # flatted로 784개를 한줄로 쭉 늘리기
flatted_x_test = x_test.reshape(-1,28*28)
print(flatted_x_train.shape)
print(flatted_x_test.shape)

fit_hist = autoencoder.fit(flatted_x_train,flatted_x_train,epochs = 100 , batch_size = 256,
                                validation_data = (flatted_x_test, flatted_x_test))


decoded_img = autoencoder.predict(flatted_x_test[:10])

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10, i+1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i+ 1 + n)
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()

