import gc
import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger, TensorBoard, LambdaCallback, Callback
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model

from .build_boost_model import (build_boost_model, get_geo_range,
                                   select_images_in_range, build_inputs)
from .metrics import m_iou, m_iou_0, m_iou_1
from .multi_gpu import MultiGPUModelCheckpoint

class Save_learning_rate(Callback):
    def __init__(self, epochs, name, num_stages, checkpoint_dir):
        self.epochs = epochs
        self.name = name
        self.num_stages  = num_stages
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, num_stages, logs={}):
        Adaptive_lr = []
        for i in range(self.num_stages - 1):
            stage_i = self.name.get_layer(name=f'stage_{i}_Adaptive_learning_rate').get_weights()[0]
            stage_i = stage_i.tolist()
            Adaptive_lr.append(stage_i[0][0])
        with open(os.path.join(self.checkpoint_dir, 'learning_rate.log'), 'a', encoding='utf-8') as f:
            f.write(f'epoch_{epoch} '+f'{Adaptive_lr}\n')

def train_boost_model(segmentation_model,
                         boost_lr,
                         DataLoader,
                         path_to_train_image,
                         path_to_train_labels,
                         train_file_names_json_file_list,
                         path_to_valid_image,
                         path_to_valid_labels,
                         num_gpu=0,
	                     workers=1,
                         batch_size=1,
                         learning_rate=3e-4,
                         checkpoint_dir='check_point',
                         weights_path_list=None,
                         base_learners_per_region =1,
                         geo_range_list=None,
                         num_epochs=2000,
                         train_input_size=(224, 224),
                         train_input_stride=(224, 224),
                         valid_input_size=(224, 224),
                         valid_input_stride=(224, 224),
                         number_of_class=2,
                         class_weight=None,
                         geohash_precision=None,
                         loss_weights=None,
                         custom_callback=None,
                         custom_loss=None,
                         use_previous_weights=True,
                         is_geoboost =True,
                         use_previous_data=True):
    train_file_names_list = []
    for json_file_name in train_file_names_json_file_list:
        with open(json_file_name, encoding='utf-8') as f:
            train_file_names = json.load(f)
            train_file_names_list.append(train_file_names)

    # if geo_range_list is None:
    #     geo_range_list = []
    #     for stage_i, item in enumerate(train_file_names_list):
    #         geo_range = get_geo_range(item)
    #         if stage_i == 0 or (not is_geoboost):
    #             geo_range = [32768, 0, 32768, 0]
    #         geo_range_list=geo_range_list+[geo_range]*base_learners_per_region
    #
    #         with open(f'region{stage_i}_geo_range.json', 'w',
    #                   encoding='utf-8') as f:
    #             geo_range_string = json.dumps(geo_range)
    #             f.write(geo_range_string)

    train_file_names_list_temp = []
    for item in train_file_names_list:
        train_file_names_list_temp = train_file_names_list_temp + [
            item
        ] * base_learners_per_region
    train_file_names_list = train_file_names_list_temp

    if weights_path_list is None:
        weights_path_list = [None] * len(train_file_names_list)


    selected_train_file_names_list = []
    for i, train_file_names in enumerate(train_file_names_list):
        selected_train_file_names = train_file_names

        if use_previous_data:
            for item in train_file_names_list[:i]:
                select_images = select_images_in_range(geo_range_list[i], item)
                selected_train_file_names = selected_train_file_names + select_images
            #remove the duplicated elements
            selected_train_file_names = set(selected_train_file_names)
            selected_train_file_names = list(selected_train_file_names)

        selected_train_file_names_list.append(selected_train_file_names)

    num_stages = len(train_file_names_list)

    stage_weights_path_list = []
    stage_train_file_names_list = []

    stage_geo_range_list = []

    for stage_i in range(num_stages):

        stage_weights_path_list.append(weights_path_list[stage_i])
        stage_train_file_names_list = selected_train_file_names_list[stage_i]
        geo_range_stage_i = [32768, 0, 32768, 0]
        stage_geo_range_list.append(geo_range_stage_i)
        # stage_geo_range_list.append(geo_range_list[stage_i])
        stage_custom_callback = custom_callback[stage_i]

        if (weights_path_list[stage_i] is None):
            print(f'Training in stage {stage_i}:')
            stage_checkpoint_dir = os.path.join(f'stage_{stage_i}',
                                                checkpoint_dir)
            train_boost_base_model(
                segmentation_model,
                boost_lr,
                DataLoader,
                path_to_train_image,
                path_to_train_labels,
                stage_train_file_names_list,
                path_to_valid_image,
                path_to_valid_labels,
                num_gpu=num_gpu,
                workers=workers,
                batch_size=batch_size,
                learning_rate=learning_rate,
                checkpoint_dir=stage_checkpoint_dir,
                weights_path_list=stage_weights_path_list,
                geo_range_list=stage_geo_range_list,
                num_epochs=num_epochs,
                train_input_size=train_input_size,
                train_input_stride=train_input_stride,
                valid_input_size=valid_input_size,
                valid_input_stride=valid_input_stride,
                number_of_class=number_of_class,
                class_weight=class_weight,
                geohash_precision=geohash_precision,
                loss_weights=loss_weights,
                custom_callback=stage_custom_callback,
                custom_loss=custom_loss,
                use_previous_weights=use_previous_weights)


def train_boost_base_model(segmentation_model,
                              boost_lr,
                              DataLoader,
                              path_to_train_image,
                              path_to_train_labels,
                              train_file_names_list,
                              path_to_valid_image,
                              path_to_valid_labels,
                              num_gpu=0,
                              workers=30,
                              batch_size=8,
                              learning_rate=3e-4,
                              checkpoint_dir='check_point',
                              weights_path_list=None,
                              geo_range_list=None,
                              num_epochs=2000,
                              train_input_size=(224, 224),
                              train_input_stride=(224, 224),
                              valid_input_size=(224, 224),
                              valid_input_stride=(224, 224),
                              number_of_class=2,
                              class_weight=None,
                              geohash_precision=None,
                              loss_weights=None,
                              custom_callback=None,
                              custom_loss=None,
                              use_previous_weights=True):

    train_generator_params = {
        'x_set_dir': path_to_train_image,
        'y_set_dir': path_to_train_labels,
        'patch_size': train_input_size,
        'patch_stride': train_input_stride,
        'batch_size': batch_size,
        'shuffle': True,
        'is_train': True,
        'file_names': train_file_names_list
    }

    test_generator_params = {
        'x_set_dir': path_to_valid_image,
        'y_set_dir': path_to_valid_labels,
        'patch_size': valid_input_size,
        'patch_stride': valid_input_stride,
        'batch_size': batch_size
    }

    if not (geohash_precision is None):
        train_generator_params['geohash_precision'] = geohash_precision
        test_generator_params['geohash_precision'] = geohash_precision

    boost_lr_list = [boost_lr] * len(weights_path_list)


    
    network_inputs = build_inputs(number_of_class=number_of_class,
                                    train_input_size=train_input_size,
                                    geohash_precision=geohash_precision)

    if num_gpu == 1:
        model, current_base_model = build_boost_model(
            segmentation_model, weights_path_list, geo_range_list,
            boost_lr_list, network_inputs, number_of_class,
            use_previous_weights=use_previous_weights)
        model = Model(inputs=network_inputs, outputs=model)
        print("Training using one GPU..")
    else:
        with tf.device('/cpu:0'):
            model, current_base_model = build_boost_model(
                segmentation_model, weights_path_list, geo_range_list,
                boost_lr_list, network_inputs, number_of_class,
                use_previous_weights=use_previous_weights)
            model = Model(inputs=network_inputs, outputs=model)

    #if not (weights_path is None):
    #model.load_weights(weights_path, by_name=True)
    if num_gpu > 1:
        parallel_model = multi_gpu_model(model, gpus=num_gpu)
        print("Training using multiple GPUs..")
    else:
        parallel_model = model
        print("Training using one GPU or CPU..")

    #if not (weights_path is None):
        #parallel_model.load_weights(weights_path, by_name=True)

    # Define a loss function to ignore pixels with a tag value of 2
    def mycrossentropy(y_true, y_pred, e=0.1):
        mask0 = K.zeros_like(y_true[:, :, 0])
        mask1 = K.ones_like(y_true[:, :, 0])
        condition = tf.where(K.equal(y_true[:, :, 2], K.variable(1)), mask0, mask1)
        return K.categorical_crossentropy(y_true[:, :, 0:2], y_pred) * condition

    if custom_loss is None:
        training_loss = mycrossentropy
    else:
        training_loss = custom_loss

    #if custom_loss is None:
        #training_loss = 'categorical_crossentropy'
    #else:
        #training_loss = custom_loss

    parallel_model.compile(loss=training_loss,
                           optimizer=keras.optimizers.Adam(learning_rate),
                           metrics=["accuracy", m_iou, m_iou_0, m_iou_1],
                           loss_weights=loss_weights)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    num_stages = len(weights_path_list)

    save_lr_callback = Save_learning_rate(num_stages = num_stages,epochs = num_epochs, name = parallel_model, checkpoint_dir = checkpoint_dir)

    tensor_boarder = TensorBoard(log_dir=checkpoint_dir, update_freq='epoch')
    csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training.log'))

    checkpointer = MultiGPUModelCheckpoint(cpu_model=current_base_model,
                                           filepath=os.path.join(
                                               checkpoint_dir, 'weights_{epoch:03d}.hdf5'),
                                           verbose=1,
                                           monitor='val_acc',
                                           mode='max',
                                           save_best_only=False,
                                           save_weights_only=True,
                                           weights_path_list = weights_path_list
                                           )
    call_back_list = [tensor_boarder, checkpointer, csv_logger, save_lr_callback]
    if not (custom_callback is None):
        call_back_list.append(custom_callback)

    train_generator = DataLoader(**train_generator_params)
    test_generator = DataLoader(**test_generator_params)

    parallel_model.fit_generator(train_generator,
                                 epochs=num_epochs,
                                 workers=workers,
                                 #workers=1,
                                 verbose=1,
                                 use_multiprocessing=True,
                                 validation_data=test_generator,
                                 max_queue_size=workers,
                                 callbacks=call_back_list,
                                 class_weight=class_weight)



    K.clear_session()
