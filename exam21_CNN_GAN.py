import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_out'
img_shape = (28, 28, 1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)

X_train = X_train / 250
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

generator = Sequential()
generator.add(Dense(256*7*7, input_dim=noise))
generator.add(Reshape((7, 7, 256))) #Reshape을 또 감싸는 이유 =  튜플임
generator.add(Conv2DTranspose(128, kernel_size=3,
            strides=2, padding='same')) #strides은 2칸씩 넘어주기 그래서 사이즈 작아짐
#Conv2DTranspose를 이용해서 256장이 128장으로 합침 즉 디코더 형식
generator.add(BatchNormalization()) #BatchNormalization은 많은 이미지를 평균이 0 표준편차를 1로 이유는? 숫자가 너무 커지니까
generator.add(LeakyReLU(alpha=0.01))

generator.add(Conv2DTranspose(64, kernel_size=3,
            strides=1, padding='same'))
generator.add(BatchNormalization())
generator.add(LeakyReLU(alpha=0.01))

generator.add(Conv2DTranspose(1, kernel_size=3,
                    strides=2, padding='same')) #Conv2DTranspose 1인 이유가 결국은 사진은 1장 나와야함
generator.add(Activation('tanh'))

#판별자
discriminator = Sequential()
discriminator.add(Conv2D(32, kernel_size=3,
        strides=2, padding='same', input_shape=img_shape)) #Conv2D 생성자와 다른이유 : 인코더 형식임
discriminator.add(LeakyReLU(alpha=0.01))
# 한층 더쌓기
discriminator.add(Conv2D(64, kernel_size=3,
        strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))
# 한층 더쌓기 2
discriminator.add(Conv2D(128, kernel_size=3,
        strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))

discriminator.add(Conv2D(256, kernel_size=3,
        strides=2, padding='same'))
discriminator.add(LeakyReLU(alpha=0.01))

discriminator.add(Flatten()) # 이미지 펼쳐주고
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
discriminator.trainable=False

gan_model = Sequential()
gan_model.add(generator)
gan_model.add(discriminator)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size, 1))

fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0, 1, (batch_size, noise))
    fake_imgs = generator.predict(z)

    d_hist_real = discriminator.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator.trainable=False

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)

    if epoch % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(
            epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col),
                             sharey=True, sharex=True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cont += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(epoch+1))
        plt.savefig(path)
        plt.close()