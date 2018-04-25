import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 6
epsilon_std = 1.0


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

vae.save("/media/petrichor/data/future/autoencoders/visualizations/weights/smallVAE")
encoder.save("/media/petrichor/data/future/autoencoders/visualizations/weights/smallencoderVAE")
# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()


randomimages = x_test[:10000]
labels = y_test[:10000]

input = randomimages.reshape(10000,784)
encoded = encoder.predict(input)
encoded = encoded.reshape(10000,2)

# encodedVAE = encoderVAE.predict(input)
# encodedVAE= encodedVAE.reshape(1000,2)

# matplotlib

# fig, axes = plt.subplots(2, sharex = True)


# colors = ['red','green','blue','purple','orange','black','pink','yellow','grey','cyan']
# axes[0].scatter(-1*encoded[:,0], encoded[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
# axes[1].scatter(-1*encodedVAE[:,0], encodedVAE[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))

# plt.show()
labels=np.array([labels])
from ggplot import *
import pandas
a=np.concatenate((encoded,labels.T), axis=1)
df= pandas.DataFrame(a,columns=['x','y','labels'])
df.to_csv("weights/vaeout.csv")
print type(df['labels'][0])
print df['labels'][0], df['labels'][100], df['labels'][10], df['labels'][11]
print df.shape
g = ggplot(df, aes(x='x', y='y',colour = 'labels')) + \
    geom_point(size=40, alpha=.4) 
print g