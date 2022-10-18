import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib  # pylint:disable = E0611


def set_keras_config():
    K.clear_session()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True  #pylint:disable = E1101

    sess = tf.Session(config=config)
    K.set_session(sess)


def get_available_gpus_num():
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpus = len(gpu_list)
    return num_gpus
