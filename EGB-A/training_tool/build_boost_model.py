from keras import backend as K
from keras import layers, models
from keras.layers import Layer

class AdaptiveLrLayer(Layer):
    def __init__(self, **kwargs):
        super(AdaptiveLrLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='Adaptive_learning_rate',
                                      shape=(1, 1),
                                      initializer='Constant',
                                      trainable=True)
        super(AdaptiveLrLayer, self).build(input_shape)

    def call(self, x):
        return (self.kernel*x)

def build_boost_model(segmentation_model, weights_path_list, geo_range_list,
                         boost_lr_list, network_inputs, number_of_class,use_previous_weights=True):

    base_models_list = []
    current_base_model = None
    for stage_index,(weights_path, geo_range, boost_lr) in enumerate(zip(weights_path_list,
                                                 geo_range_list,
                                                 boost_lr_list)):
        model = segmentation_model(network_inputs,
                                   number_of_class=number_of_class,
                                   geo_range=geo_range)
        model = models.Model(network_inputs, model)
        if not (weights_path is None):
            model.load_weights(weights_path, by_name=True)
            for layer in model.layers:
                layer.trainable = False
                if not isinstance(layer,layers.InputLayer):                   
                    layer.name = f"{layer.name}_stage_{stage_index}"
    
            x = model.output
            model_output = AdaptiveLrLayer(name=f'stage_{stage_index}_Adaptive_learning_rate')(x)

            # #Artificial learning rate of EGB
            # def multiply_boost_lr_(x):
            #    #boost_lr = K.constant(boost_lr, dtype='float32')
            #    model_output = boost_lr * x
            #    return model_output
            # model_output = layers.Lambda(multiply_boost_lr_)(x)
                       
        else:
            if use_previous_weights and (stage_index>0):
                model.load_weights(weights_path_list[0], by_name=True)
            model_output = model.output

        base_models_list.append(model_output)
        current_base_model = model

    if len(weights_path_list) > 1:
        boost_model = layers.add(base_models_list)
    else:
        boost_model = base_models_list[0]

    x = layers.Activation('softmax')(boost_model)

    return x, current_base_model


def get_geo_range(image_file_name_list):

    geo_range = [1,1,1,1]

    return geo_range

    # # GeoBoost: get the range of images by file name
    # image_x_list = []
    # image_y_list = []
    # for item in image_file_name_list:
    #     x = item.split('_')[3]
    #     y = item.split('_')[5]
    #
    #     image_x_list.append(int(x))
    #     image_y_list.append(int(y))
    #
    # geo_range = []
    # geo_range.append(max(image_x_list))
    # geo_range.append(min(image_x_list))
    # geo_range.append(max(image_y_list))
    # geo_range.append(min(image_y_list))
    #
    # return geo_range


def select_images_in_range(geo_range, image_file_name_list):

    select_images_list = image_file_name_list

    return select_images_list

    # # GeoBoost: select images in a specific region
    # select_images_list = []
    # for item in image_file_name_list:
    #     x = item.split('_')[3]
    #     y = item.split('_')[5]
    #
    #     image_x = int(x)
    #     image_y = int(y)
    #
    #     if (image_x <= geo_range[0]) and (image_x >= geo_range[1]) and (
    #             image_y <= geo_range[2]) and (image_y >= geo_range[3]):
    #
    #         select_images_list.append(item)
    #
    # return select_images_list


def build_inputs(number_of_class,train_input_size,geohash_precision=None):
    input_img_shape = (*train_input_size, 3)
    img_input = layers.Input(shape=input_img_shape)
    network_inputs = [img_input]

    if not (geohash_precision is None):

        input_geohash_shape = (1, 1, geohash_precision)
        geohash_input = layers.Input(shape=input_geohash_shape)
        network_inputs.append(geohash_input)

    patch_x_input = layers.Input(shape=(1, ))
    network_inputs.append(patch_x_input)
    patch_y_input = layers.Input(shape=(1, ))
    network_inputs.append(patch_y_input)

    return network_inputs
