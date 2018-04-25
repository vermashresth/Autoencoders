
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

(X, y_train), (x_test, y_test) = mnist.load_data()

X = X.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
X= X.reshape(60000,28,28,1)


encoder = load_model("weights/encoder")
model = load_model("weights/model")


randomimages = x_test[:1000]
labels = y_test[:1000]

input = randomimages.reshape(1000,28,28,1)
encoded = encoder.predict(input)
encoded = encoded.reshape(1000,2)


fig, axes = plt.subplots(2, sharex = True)


colors = ['red','green','blue','purple','orange','black','pink','yellow','grey','cyan']
axes[1].scatter(-1*encoded[:,0], encoded[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
plt.show()