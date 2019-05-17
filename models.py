import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, GlobalAveragePooling2D
from keras import regularizers
from keras.activations import relu, softmax
from keras.layers.merge import add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

image_dim = 28
num_classes = 10
num_channels = 1


def alex_net():
    model = Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same',
                     input_shape=(image_dim, image_dim, num_channels,)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html
def alex_net_original():
    model = Sequential()

    # Layer 1
    model.add(Conv2D(96, 11, 11, input_shape=(28, 28, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(256, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    # Layer 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(1024, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Flatten())
    model.add(Dense(3072, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 7
    model.add(Dense(4096, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Layer 8
    model.add(Dense(10, init='glorot_normal'))
    model.add(Activation('softmax'))

    # (4) Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# https://www.kaggle.com/ashishpatel26/mnist-alexnet-using-keras
def lenet5():
    model = Sequential()
    # Layer 1
    # Conv Layer 1
    model.add(Conv2D(filters=6,
                     kernel_size=5,
                     strides=1,
                     activation='relu',
                     input_shape=(image_dim, image_dim, 1)))
    # Pooling layer 1
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 2
    # Conv Layer 2
    model.add(Conv2D(filters = 16,
                     kernel_size = 5,
                     strides = 1,
                     activation = 'relu',
                     input_shape = (14,14,6)))
    #Pooling Layer 2
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    #Flatten
    model.add(Flatten())
    #Layer 3
    #Fully connected layer 1
    model.add(Dense(units = 120, activation = 'relu'))
    #Layer 4
    #Fully connected layer 2
    model.add(Dense(units = 84, activation = 'relu'))
    #Layer 5
    #Output Layer
    model.add(Dense(units = num_classes, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model


# https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
def block(n_output, upscale=False):
	# n_output: number of feature maps in the block
	# upscale: should we use the 1x1 Conv2D mapping for shortcut or not

	# keras functional api: return the function of type
	# Tensor -> Tensor
	def f(x):

	    # H_l(x):
	    # first pre-activation
	    h = BatchNormalization()(x)
	    h = Activation(relu)(h)
	    # first convolution
	    h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

	    # second pre-activation
	    h = BatchNormalization()(x)
	    h = Activation(relu)(h)
	    # second convolution
	    h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

	    # f(x):
	    if upscale:
	        # 1x1 Conv2D
	        f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
	    else:
	        # identity
	        f = x

	    # F_l(x) = f(x) + H_l(x):
	    return add([f, h])

	return f

# https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
def tiny_resnet():
    # input tensor is the 28x28 grayscale image
    input_tensor = Input((image_dim, image_dim, 1))

    # first Conv2D with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # F_1
    x = block(16)(x)
    # F_2
    x = block(16)(x)

    # F_3
    # H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32
    # and we can't add together tensors of inconsistent sizes, so we use upscale=True
    # x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_4
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation
    # F_5
    # x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

    # F_6
    # x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation
    # F_7
    # x = block(48)(x)                     # !!! <------- Uncomment for local evaluation

    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)

    # dropout for more robust learning
    x = Dropout(0.2)(x)

    # last softmax layer
    x = Dense(units=num_classes, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation(softmax)(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html
# standard = relu last is softmax without dropout
def batch_norm():
    model = Sequential()

    model.add(Conv2D(6, 5, 5, border_mode='valid', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(16, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(120, 1, 1, border_mode='valid'))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

    return model


# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html
# relu standard last is softmax
def pure_conv():
    model = Sequential()

    model.add(Conv2D(96, 5, 5, border_mode='valid', input_shape = (28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(Activation("relu"))

    model.add(Conv2D(192, 5, 5, border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(Activation("relu"))

    model.add(Conv2D(192, 3, 3, border_mode='valid'))
    model.add(Activation("relu"))
    model.add(Conv2D(192, 1, 1, border_mode='valid'))
    model.add(Activation("relu"))
    model.add(Conv2D(10, 1, 1, border_mode='valid'))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# http://euler.stat.yale.edu/~tba3/stat665/lectures/lec18/notebook18.html
# def googlenet():
# 	model = Graph()
# 	model.add_input(name='n00', input_shape=(1,28,28))

# 	# layer 1
# 	model.add_node(Conv2D(64,1,1, activation='relu'), name='n11', input='n00')
# 	model.add_node(Flatten(), name='n11_f', input='n11')

# 	model.add_node(Conv2D(96,1,1, activation='relu'), name='n12', input='n00')

# 	model.add_node(Conv2D(16,1,1, activation='relu'), name='n13', input='n00')

# 	model.add_node(MaxPooling2D((3,3),strides=(2,2)), name='n14', input='n00')

# 	# layer 2
# 	model.add_node(Conv2D(128,3,3, activation='relu'), name='n22', input='n12')
# 	model.add_node(Flatten(), name='n22_f', input='n22')

# 	model.add_node(Conv2D(32,5,5, activation='relu'), name='n23', input='n13')
# 	model.add_node(Flatten(), name='n23_f', input='n23')

# 	model.add_node(Conv2D(32,1,1, activation='relu'), name='n24', input='n14')
# 	model.add_node(Flatten(), name='n24_f', input='n24')

# 	# output layer
# 	model.add_node(Dense(1024, activation='relu'), name='layer4',
# 	               inputs=['n11_f', 'n22_f', 'n23_f', 'n24_f'], merge_mode='concat')
# 	model.add_node(Dense(10, activation='softmax'), name='layer5', input='layer4')
# 	model.add_output(name='output1',input='layer5')
# 	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 	return model
