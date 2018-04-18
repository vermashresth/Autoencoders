from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

(X, y_train), (x_test, y_test) = mnist.load_data()

X = X.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
X= X.reshape(60000,28,28,1)
print X.shape



inputs = Input(shape=(28, 28, 1))
# h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(inputs)
# encoded = MaxPooling2D((2, 2))(h)
# h = Conv2D(4, 3, 3, activation='relu', border_mode='same')(encoded)
# h = UpSampling2D((2, 2))(h)
# outputs = Conv2D(1, 3, 3, activation='relu', border_mode='same')(h)


x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
hidden = Flatten()(encoded)
print hidden.shape
latent_size = 5
mean = Dense(latent_size)(hidden)
print mean.shape
# we usually don't directly compute the stddev sig
# but the log of the stddev instead, which is log(sig)
# the reasoning is similar to why we use softmax, instead of directly outputting
# numbers in fixed range [0, 1], the network can output a wider range of numbers which we can later compress down
log_stddev = Dense(latent_size)(hidden)

def sampling(args):
    mean, log_stddev = args
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    std_norm = K.random_normal(shape=(K.shape(mean)[0], latent_size), mean=0, stddev=1)
    # sampling from Z~N(mu, sig^2) is the same as sampling from mu + sigX, X~N(0,1)
    return mean + K.exp(log_stddev) * std_norm
  
latent_vector = Lambda(sampling)([mean, log_stddev])
print latent_vector.shape
# latent_vector = latent_vector.reshape()
x = Conv2D(8, (3, 3), activation='relu', padding='same')(latent_vector)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

def vae_loss(input_img, output):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(output-input_img))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    # return the average loss over all images in batch
    total_loss = K.mean(reconstruction_loss + kl_loss)    
    return total_loss


model = Model(input=inputs, output=decoded)
latent_encoder = Model(inputs, latent_vector)
model.compile(optimizer='adam', loss=vae_loss)
print model.summary()
model.fit(X, X, batch_size=64, nb_epoch=1)


model.save("/media/petrichor/data/future/autoencoders/varmodel")
latent_encoder.save("/media/petrichor/data/future/autoencoders/varencoder")

decoded_imgs = model.predict(x_test.reshape(10000,28,28,1))
encoded_imgs =latent_encoder.predict(x_test.reshape(10000,28,28,1))

def show_imgs(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(28,28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

show_imgs(x_test, decoded_imgs)

def show_encoding(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(4,196))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+ 1 +n)
            plt.imshow(decoded_imgs[i].reshape(4,196))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

plt.imshow(encoded_imgs[0].reshape(4,32))
plt.show()