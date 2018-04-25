
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
import numpy as np

(X, y_train), (x_test, y_test) = mnist.load_data()

X = X.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
X= X.reshape(60000,28,28,1)


encoder = load_model("weights/encodercnnAE")
model = load_model("weights/modelcnnAE")

# modelVAE = load_model("weights/smallVAE")
# encoderVAE = load_model("weights/smallencoderAE")


randomimages = x_test[:10000]
labels = y_test[:10000]

input = randomimages.reshape(10000,28,28,1)
# encoded = encoder.predict(input)
# encoded = encoded.reshape(10000,2)

# encodedVAE = encoderVAE.predict(input)
# encodedVAE= encodedVAE.reshape(1000,2)

# matplotlib

# fig, axes = plt.subplots(2, sharex = True)


# colors = ['red','green','blue','purple','orange','black','pink','yellow','grey','cyan']
# axes[0].scatter(-1*encoded[:,0], encoded[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))
# axes[1].scatter(-1*encodedVAE[:,0], encodedVAE[:,1], c = labels, cmap = matplotlib.colors.ListedColormap(colors))

# plt.show()
# labels=np.array([labels])
from ggplot import *
import pandas
# a=np.concatenate((encoded,labels.T), axis=1)
# df= pandas.DataFrame(a,columns=['x','y','labels'])

# print type(df['labels'][0])
# print df['labels'][0], df['labels'][100], df['labels'][10], df['labels'][11]
# print df.shape
# g = ggplot(df, aes(x='x', y='y',colour = 'labels')) + \
#     geom_point(size=40, alpha=.4) 
#print g


df2 = pandas.read_csv("weights/vaeout.csv")
g = ggplot(df2, aes(x='x', y='y',color = 'labels')) + \
    geom_point(size=40, alpha=.6) + \
    scale_color_gradient(low='#c8e234',mid ='#47ccbe', high='#cc4780')
print g

df3 = pandas.read_csv("weights/ae.csv")
g = ggplot(df3, aes(x='x', y='y',color = 'labels')) + \
    geom_point(size=40, alpha=.6)+ \
    scale_color_gradient(low='#c8e234',mid ='#47ccbe', high='#cc4780')
print g