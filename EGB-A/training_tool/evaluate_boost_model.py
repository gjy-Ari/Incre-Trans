import gc
import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger, TensorBoard
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model

from .build_boost_model import (build_boost_model, get_geo_range,select_images_in_range, build_inputs)
from .metrics import m_iou, m_iou_0, m_iou_1

def evaluate_boost_model(segmentation_model,
                              file_name,
                              boost_lr,
                              DataLoader,
                              test_file_names_list,
                              path_to_valid_image,
                              path_to_valid_labels,
                              num_gpu=0,
                              workers=30,
                              batch_size=1,
                              learning_rate=3e-4,
                              weights_path_list=None,
                              geo_range_list=None,
                              valid_input_size=(224, 224),
                              valid_input_stride=(224, 224),
                              number_of_class=2,
                              class_weight=None,
                              geohash_precision=None,
                              loss_weights=None,
                              custom_loss=None):


    test_generator_params = {
        'x_set_dir': path_to_valid_image,
        'y_set_dir': path_to_valid_labels,
        'patch_size': valid_input_size,
        'patch_stride': valid_input_stride,
        'batch_size': batch_size,
        'file_names': test_file_names_list
    }

    if not (geohash_precision is None):
        train_generator_params['geohash_precision'] = geohash_precision
        test_generator_params['geohash_precision'] = geohash_precision

    boost_lr_list = [boost_lr] * len(weights_path_list)


    
    network_inputs = build_inputs(number_of_class=number_of_class,
                                  train_input_size=valid_input_size,
                                  geohash_precision=geohash_precision)

    if num_gpu == 1:
        model, current_base_model = build_boost_model(
            segmentation_model, weights_path_list, geo_range_list,
            boost_lr_list, network_inputs, number_of_class)
        model = Model(inputs=network_inputs, outputs=model)
        print("Training using one GPU..")
    else:
        with tf.device('/cpu:0'):
            model, current_base_model = build_boost_model(
                segmentation_model, weights_path_list, geo_range_list,
                boost_lr_list, network_inputs, number_of_class)
            model = Model(inputs=network_inputs, outputs=model)

    if num_gpu > 1:
        parallel_model = multi_gpu_model(model, gpus=num_gpu)
        print("Training using multiple GPUs..")
    else:
        parallel_model = model
        print("Training using one GPU or CPU..")

    # if not (weights_path is None):
    #     parallel_model.load_weights(weights_path, by_name=True)

    if custom_loss is None:
        training_loss = 'categorical_crossentropy'
    else:
        training_loss = custom_loss

    parallel_model.compile(loss=training_loss,
                           optimizer=keras.optimizers.Adam(learning_rate),
                           metrics=["accuracy", m_iou, m_iou_0, m_iou_1],
                           loss_weights=loss_weights)

    test_generator = DataLoader(**test_generator_params)

    score = parallel_model.evaluate_generator(test_generator,
                                    verbose=1,
                                    workers=workers,
                                    use_multiprocessing=True)
    
    with open(file_name,"w") as f:
        f.writelines(str(score))
    K.clear_session()

