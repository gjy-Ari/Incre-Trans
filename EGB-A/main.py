import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

import sys
root_folder = ' '
sys.path.append(root_folder)
from net_architecture.nasnet_4x_geohash_SE_mobile_no_softmax import NAS_U_Net  #pylint:disable = E0401
from training_tool.train_boost_model import train_boost_model  #pylint:disable = E0401
from training_tool.loss_function import dice_coef_loss  #pylint:disable = E0401
from data_loader.dataloader_geohash_boost_nasnet import InriaDataLoaderGeohashBoostNASNet  #pylint:disable = E0401

from utils.keras_config import set_keras_config, get_available_gpus_num  #pylint:disable = E0401,E0611
from training_tool.lr_tricks import LearningRateFinder, CyclicCosineRestart  #pylint:disable = E0401,E0611

dataset_path = ' '
split_train_image_folder = os.path.join(dataset_path, 'train_image')
split_train_label_folder = os.path.join(dataset_path, 'train_label')
split_valid_image_folder = os.path.join(dataset_path, 'test_image')
split_valid_label_folder = os.path.join(dataset_path, 'test_label')

path_to_train_image = split_train_image_folder
path_to_train_labels = split_train_label_folder
path_to_valid_image = split_valid_image_folder
path_to_valid_labels = split_valid_label_folder

file_name_folder = '/home/file_name'
stage0_file_names = os.path.join(file_name_folder, 'DREAM_B.json')
stage1_file_names = os.path.join(file_name_folder, 'xBD.json')
stage2_file_names = os.path.join(file_name_folder, 'xBD_cycle.json')

base_learners_per_region = 1

train_file_names_json_file_list = [
    stage0_file_names, stage1_file_names, stage2_file_names
]

# lr initialized to 0.5
boost_lr = 0.5
use_previous_weights = True
use_previous_data = False

# When training the nth base learner, specify the model of the previous n-1 base learners. Here n=3.
weights_path_list = ['stage_0/check_point/weights_100.hdf5', 'stage_1/check_point/weights_100.hdf5']+[None]*1
optimizer_lr_list = [3e-4, 3e-4, 3e-4] * base_learners_per_region

lr_callback_list = []
for optimizer_lr in optimizer_lr_list:
    lr_callback = CyclicCosineRestart(lr_min=1e-6,
                                      lr_max=optimizer_lr,
                                      number_of_lr_warm_epochs=10,
                                      number_of_epochs=30,
                                      use_warmup=True)

    lr_callback_list.append(lr_callback)

train_boost_model(NAS_U_Net,
                     boost_lr,
                     InriaDataLoaderGeohashBoostNASNet,
                     path_to_train_image,
                     path_to_train_labels,
                     train_file_names_json_file_list,
                     path_to_valid_image,
                     path_to_valid_labels,
                     # num_gpu=1,
                     num_gpu=get_available_gpus_num(),
                     workers=10,
                     batch_size=48,
                     learning_rate=3e-4,
                     checkpoint_dir='check_point',
                     weights_path_list=weights_path_list,
                     base_learners_per_region=base_learners_per_region,
                     geo_range_list=None,
                     num_epochs=30,
                     train_input_size=(512, 512),
                     train_input_stride=(512, 512),
                     valid_input_size=(512, 512),
                     valid_input_stride=(512, 512),
                     number_of_class=2,
                     class_weight=None,
                     geohash_precision=None,
                     loss_weights=None,
                     custom_callback=lr_callback_list,
                     custom_loss=None,
                     use_previous_weights=use_previous_weights,
                     use_previous_data=use_previous_data)
