import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential


OUT_DIR = './DNN_out'
img_shape = (28, 28, 1) # 28 x 28 = 728사진을 3차원으로 만듬
epochs = 100000 # 학습이 10만번
batch_size = 128
noise = 100 # 노이즈가 100이지만 랜덤값으로 줄거임
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data() # 간 모델이기때문에 X_train만 있으면 됨
print(X_train.shape)

X_train = X_train / 127.5 - 1 # /250하면 0~1임 -1부터 1까지 범위를 만듬
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

# 생성자
generator = Sequential()
generator.add(Dense(128, input_dim=noise)) #
generator.add(LeakyReLU(alpha=0.1)) #그리고 LeakyReLU를 사용하면 -값을 조금 살림
generator.add(Dense(784, activation='tanh')) # 값 자체가 학습데이터가 -1~ 1이기 때문에 tanh으로 나오기
generator.add(Reshape(img_shape))
generator.summary()

# 판별자
lrelu = LeakyReLU(alpha=0.1) #tanh를 사용했기 때문에 LeakyReLU를 사용함
discriminator = Sequential()
discriminator.add(Flatten(input_shape=img_shape)) # Flatten을 이용한 사진값 쭉 펼치기
discriminator.add(Dense(128, activation = lrelu))
discriminator.add(Dense(1, activation = 'sigmoid')) # 맞으면 1 아니면 0이기 때문에 시그모이드임
discriminator.summary()

discriminator.compile(loss = 'binary_crossentropy',
                      optimizer='adam', metrics= ['accuracy'])

gen_model = Sequential()
gen_model.add(generator)
gen_model.add(discriminator)
gen_model.summary()
gen_model.compile(loss = 'binary_crossentropy', optimizer='adam')

real = np.ones((batch_size,1)) # 사실이면 1

fake = np.zeros((batch_size,1)) # 거짓이면 0

discriminator.trainable = False # 판별자만 먼저 학습을 안시키는 코드

for epoch in range(epochs):
    idx = np.random.randint(0,X_train.shape[0], batch_size) # 0부터 6만개까지 128개까지 부트스트랩이라고 한다
    # 중복데이터를 삭제 안하는 이유 사람키가 170인사람이 여려 있을수 있음
    real_imgs = X_train[idx] # 진짜 이미지를 X_train에 넣음
    z = np.random.normal(0, 1, (batch_size,noise)) # 평균이 0이고 표준편차가 1인거
    fake_imgs = generator.predict(z) # 가짜 이미지를 생성자에 집어넣음

    d_hist_real = discriminator.train_on_batch(real_imgs, real) # train_on_batch 데이터 한묶음 주고 학습 다 하면 끝
    d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)


    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # 진짜와 가짜의 평균
    discriminator.trainable = False # 다시 한번 밸런스를 위해서 판별자 학습을 중단시킴

    if epoch %4 == 0: # 생성자 4번 학습시키면 판별자를 1번 학습시킴
        z_gan = np.random.normal(0, 1, (batch_size, noise))  # 평균이 0이고 표준편차가 1인거
        gan_hist = gen_model.train_on_batch(z_gan,real)

    if epoch % sample_interval == 0:
        print('%d [D loss %f, acc.: %.2f%%] [G loss: %f]'%(
            epoch, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0,1,(row * col, noise))
        fake_imgs = generator.predict(z_gan)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col),
                             sharey = True, sharex = True)
        cont = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cont, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cont +=1
        path = os.path.join(OUT_DIR, 'img--{}'.format(epoch))
        plt.savefig(path)
        plt.close()





