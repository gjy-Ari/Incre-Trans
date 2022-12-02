import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import cv2
from keras import backend as K
import sys
# Folder location of the model
root_folder = '/home/EGB-A'
sys.path.append(root_folder)
from net_architecture.nasnet_4x_geohash_SE_mobile_no_softmax import NAS_U_Net  #pylint:disable = E0401
from training_tool.model_predict import Boost_Model  #pylint:disable = E0401
import fnmatch
from keras.applications.nasnet import preprocess_input
from utils.predict_cropimage import Cropimage

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

base_learners_per_region = 1
number_of_class = 2

# Prediction using a model with n base learners.
# The path of the data to be predicted and the path to save the prediction results.
filepaths = r'./data'
outpaths = f'./predict'

# Copy the learning rates from the 'learning_rate.log' to the previous n-1 base learners.
boost_lr = [0.2340451329946518, 0.7576878666877747]
input_size = 1024

# Specify the weights of n base learners.
weights_path_list = ['stage_0/check_point/weights_100.hdf5', 'stage_1/check_point/weights_100.hdf5', 'stage_2/check_point/weights_030.hdf5']

# stage_n = 2 represents the third stage, with 3 base learners.
stage_n = 2

# Region division is not required, each stage is the same [32768, 0, 32768, 0].
geo_range_list = []
for i in range(len(weights_path_list)):
    geo_range_list.append([32768, 0, 32768, 0])
print(geo_range_list)

boost_model = Boost_Model(NAS_U_Net,
                          weights_path_list,
                          number_of_class,
                          preprocess_input,
                          input_size,
                          boost_lr,
                          geo_range_list)
print(f'Predicting model stage{len(geo_range_json_list)-1}..')

n = 0
for item in os.listdir(filepaths):
    if item.endswith('.tif'):
        n+=1
        img = cv2.imread(os.path.join(filepaths,item))
        shape = img.shape[:2]
        crop_shape = [input_size,input_size]
        cropdir = Cropimage(os.path.join(filepaths,item),crop_shape)
        test_image_folder = os.path.join(filepaths,item.split('.')[0],f'Crop_images_{input_size}')
        file_names = [file_name for file_name in os.listdir(test_image_folder)
                    if fnmatch.fnmatch(file_name,'*.tif')]

        for i, file_name in enumerate(file_names):
                output_dir = os.path.join(outpaths, f'probability_stage{stage_n}', f'{item}')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                boost_model.predict_and_save_prediction(
                test_image_folder,
                output_dir,
                file_name)
        print(str(n)+'th image is done..')
K.clear_session()
print('predict is done')