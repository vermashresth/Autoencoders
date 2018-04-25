
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K

(X, y_train), (x_test, y_test) = mnist.load_data()

X = X.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
X= X.reshape(60000,28,28,1)

def vae_loss(input_img, output):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(output-input_img))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + log_stddev - K.square(mean) - K.square(K.exp(log_stddev)), axis=-1)
    # return the average loss over all images in batch
    total_loss = K.mean(reconstruction_loss + kl_loss)    
    return total_loss
mean=load_model("weights/meanVAE")
log_stddev = load_model("weights/stdVAE")


encoder = load_model("weights/modelVAE",custom_objects={'vae_loss':vae_loss, 'log_stddev':log_stddev, 'mean':mean})
model = load_model("weights/encoderVAE",custom_objects={'vae_loss':vae_loss, 'log_stddev':log_stddev, 'mean':mean})



decoded_imgs = model.predict(x_test.reshape(10000,28,28,1))
encoded_imgs = encoder.predict(x_test.reshape(10000,28,28,1))

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