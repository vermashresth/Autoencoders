from keras.models import load_model
from keras.datasets import mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

encoder = load_model("encoder")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

randomimages = np.random.shuffle(x_test)[:1000]

input = randomimages.reshape(1000,28,28,1)
encoded = encoder.predict(input)
print encoded.shape