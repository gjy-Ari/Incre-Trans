from keras import backend as K
from keras import layers
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Lambda, Reshape, SeparableConv2D, SpatialDropout2D,
                          UpSampling2D)
from keras.regularizers import l2

from .nasnet_keras_SE_nodropout import NASNetMobile, _normal_a_cell
from .SE_block import csSE_block

bn_momentum = 0.9997
interpolation = 'bilinear'
num_blocks = 4


def geohash_concat(x, geohash_input):
    feature_map_height, feature_map_width = K.int_shape(x)[-3:-1]
    geohash_code_feature = layers.UpSampling2D(
        size=(feature_map_height, feature_map_width),
        interpolation=interpolation)(geohash_input)

    x = layers.Concatenate(axis=-1)([x, geohash_code_feature])
    return x


def transition_up(x,
                  p,
                  num_filter,
                  block_id,
                  weight_decay=0.0,
                  geohash_input=None):
    x = UpSampling2D(size=(2, 2), interpolation=interpolation)(x)
    if not (geohash_input is None):
        x = geohash_concat(x, geohash_input)
    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, num_filter, block_id=f'{block_id}_{i}')
    x = csSE_block(x, block_id)
    return x


def NAS_U_Net(input_tensor, number_of_class, geo_range, weight_decay=0.0):

    img_input = input_tensor[0]
    base_model = NASNetMobile(include_top=False,
                              weights='imagenet',
                              input_tensor=img_input)

    normal_cell_4_x = base_model.get_layer('res_4x').output
    normal_cell_8_x = base_model.get_layer('res_8x').output

    normal_cell_16_x = base_model.get_layer('res_16x').output

    normal_cell_32_x = base_model.get_layer('res_32x').output

    penultimate_filters = 1056
    filters = penultimate_filters // 24
    filter_multiplier = 2

    x = transition_up(normal_cell_32_x,
                      normal_cell_16_x,
                      filters * filter_multiplier,
                      '32_to_16',
                      weight_decay=weight_decay)

    x = transition_up(x,
                      normal_cell_8_x,
                      filters,
                      '16_to_8',
                      weight_decay=weight_decay)

    if len(input_tensor) == 4:
        geohash_input = input_tensor[1]
    else:
        geohash_input = None

    x = transition_up(x,
                      normal_cell_4_x,
                      filters // filter_multiplier,
                      '8_to_4',
                      weight_decay=weight_decay,
                      geohash_input=geohash_input)

    x = Activation('relu')(x)

    x = Conv2D(filters=number_of_class,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=True,
               padding='same',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(weight_decay),
               name='last_conv')(x)
    x = UpSampling2D(size=(4, 4), interpolation=interpolation)(x)
    x = Reshape((-1, number_of_class))(x)

    patch_x = input_tensor[-2]
    patch_y = input_tensor[-1]

    def is_in_geo_range(patch_x_and_y):

        patch_x_tensor = patch_x_and_y[0]
        patch_y_tensor = patch_x_and_y[1]
        base_leaner_x_max = K.constant(geo_range[0], dtype='float32')
        base_leaner_x_min = K.constant(geo_range[1], dtype='float32')
        base_leaner_y_max = K.constant(geo_range[2], dtype='float32')
        base_leaner_y_min = K.constant(geo_range[3], dtype='float32')

        layer_output = patch_x_and_y[2]
        coef_geo_range = layer_output

        coef_geo_range = K.switch(K.less_equal(patch_x_tensor, base_leaner_x_max),
                                  coef_geo_range * 1.0, coef_geo_range * 0.0)

        coef_geo_range = K.switch(
            K.greater_equal(patch_x_tensor, base_leaner_x_min),
            coef_geo_range * 1.0, coef_geo_range * 0.0)

        coef_geo_range = K.switch(K.less_equal(patch_y_tensor, base_leaner_y_max),
                                  coef_geo_range * 1.0, coef_geo_range * 0.0)

        coef_geo_range = K.switch(
            K.greater_equal(patch_y_tensor, base_leaner_y_min),
            coef_geo_range * 1.0, coef_geo_range * 0.0)

        return coef_geo_range

    x = Lambda(is_in_geo_range)([patch_x, patch_y, x])

    return x
