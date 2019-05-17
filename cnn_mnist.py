import keras
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from tflearn.data_utils import image_preloader
from PIL import Image
from keras.callbacks import ModelCheckpoint
from models import tiny_resnet, lenet5, alexnet, batch_norm, pure_conv

# Global parameters
batch_size = 64
num_epochs = 50
num_classes = 10
image_dim = 28
input_shape = (image_dim, image_dim, 1)


def load_mnist():
    # Load data and reshape
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], image_dim, image_dim, 1)
    x_test = x_test.reshape(x_test.shape[0], image_dim, image_dim, 1)
    # input_shape = (image_dim, image_dim, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_mnist()

model = lenet5()

# checkpoint
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
callbacks_list = [checkpoint]

# Data augmentation
gen = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.1,
                         shear_range=0.2,
                         height_shift_range=0.1,
                         zoom_range=0.1)
test_gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

# Run model
history = model.fit_generator(train_generator,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              epochs=num_epochs,
                              validation_data=test_generator,
                              validation_steps=x_test.shape[0] // batch_size,
                              callbacks=callbacks_list,
                              class_weight='auto')


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
