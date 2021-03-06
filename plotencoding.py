from keras.models import load_model
from keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()



encoder = load_model("encoder1iter")
print x_test[0].shape
randomimages = x_test[:1000]
labels = y_test[:1000]

input = randomimages.reshape(1000,28,28,1)
encoded = encoder.predict(input)
encoded = encoded.reshape(1000,128)
print encoded.shape


pca = PCA(n_components=2)
encoded_transformed = pca.fit_transform(encoded)

print encoded_transformed.shape
print encoded_transformed[0]
print encoded_transformed[:][0], encoded_transformed[:][1]


fig, axes = plt.subplots(2, sharex = True)


colors = ['red','green','blue','purple','orange','black','pink','yellow','grey','cyan']
axes[0].scatter(encoded_transformed[:,0], encoded_transformed[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
# cb = axes[0,0].colorbar()
# loc = np.arange(0,max(labels),max(labels)/float(len(colors)))
# cb.set_ticks(loc)
# cb.set_ticklabels(colors)


encoder = load_model("encoder5iter")
print x_test[0].shape
randomimages = x_test[:1000]
labels = y_test[:1000]

input = randomimages.reshape(1000,28,28,1)
encoded = encoder.predict(input)
encoded = encoded.reshape(1000,128)
print encoded.shape


pca = PCA(n_components=2)
encoded_transformed = pca.fit_transform(encoded)

print encoded_transformed.shape
print encoded_transformed[0]
print encoded_transformed[:][0], encoded_transformed[:][1]

axes[1].scatter(-1*encoded_transformed[:,0], encoded_transformed[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
plt.show()



