from keras import backend as K
from keras.layers import Dense, Conv2D, Multiply, Add, Lambda, GlobalAveragePooling2D


def cSE_block(input_x):
    x = GlobalAveragePooling2D()(input_x)
    x = Dense(K.int_shape(input_x)[-1] // 16,
              activation='relu',
              kernel_initializer="he_normal")(x)
    x = Dense(K.int_shape(input_x)[-1],
              activation='sigmoid',
              kernel_initializer="he_normal")(x)
    x = Multiply()([input_x, x])
    return x


def sSE_block(input_x):
    x = Conv2D(1, (1, 1),
               padding="same",
               kernel_initializer="he_normal",
               activation='sigmoid',
               strides=(1, 1))(input_x)
    x = Multiply()([input_x, x])
    return x


def csSE_block(x, name):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cSE = cSE_block(x)
    sSE = sSE_block(x)
    x = Add(name=name)([cSE, sSE])
    return x