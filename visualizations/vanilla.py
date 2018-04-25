from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1

import numpy as np
import matplotlib
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
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((4, 4), padding='same')(x)
x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (1, 1, 2) i.e. 2-dimensional

x = Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)




model = Model(input=inputs, output=decoded)
encoder = Model(inputs, encoded)
model.compile(optimizer='adam', loss='binary_crossentropy')
print model.summary()


model.fit(X, X, batch_size=64, nb_epoch=4)


model.save("/media/petrichor/data/future/autoencoders/visualizations/weights/modelcnnAE")
encoder.save("/media/petrichor/data/future/autoencoders/visualizations/weights/encodercnnAE")


randomimages = x_test[:10000]
labels = y_test[:10000]

input = randomimages.reshape(10000,28,28,1)
encoded = encoder.predict(input)
encoded = encoded.reshape(10000,2)

import numpy as np
labels=np.array([labels])
import pandas
a=np.concatenate((encoded,labels.T), axis=1)
df= pandas.DataFrame(a,columns=['x','y','labels'])

df.to_csv("weights/ae.csv")


fig, axes = plt.subplots(2, sharex = True)


colors = ['red','green','blue','purple','orange','black','pink','yellow','grey','cyan']
axes[1].scatter(-1*encoded[:,0], encoded[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
plt.show()