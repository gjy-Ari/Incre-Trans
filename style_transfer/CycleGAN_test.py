import tensorflow as tf
from skimage import io
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow_examples.models.pix2pix import pix2pix
import glob
import time
import matplotlib.pyplot as plt


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize, x:x + windowSize])

def walk_type(path, file_type):
    paths = glob.glob(os.path.join(path,
                                   file_type
                                   )
                      )
    return paths


def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[2048, 2048, 3])
  return cropped_image

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  image = tf.image.resize(image, [2078, 2078],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  image = random_crop(image)
  image = tf.image.random_flip_left_right(image)

  return image
def preprocess_image_train(image):
    image_string = tf.read_file(image
                                )
    image_decoded = tf.image.decode_png(image_string, )
    image = random_jitter(image_decoded)
    image = normalize(image)
    return image

def preprocess_image_train1(image):
    image_string = tf.read_file(image
                                )
    image_decoded = tf.image.decode_jpeg(image_string, )
    image = random_jitter(image_decoded)
    image = normalize(image)
    return image
def preprocess_image_test(image):
    image_string = tf.read_file(image)
    image_decoded = tf.image.decode_png(image_string, )
    image = normalize(image_decoded)
    return image


paths4 = walk_type(r'','*.png') # image path4
filenames4 = tf.constant(paths4)
train_zebras = tf.data.Dataset.from_tensor_slices((filenames4))
train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)

paths2 = walk_type(r'','*.png') # image path2
filenames2 = tf.constant(paths2)
train_horses = tf.data.Dataset.from_tensor_slices((filenames2))
train_horses = train_horses.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(1)


OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)



LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!')


filepath = r'/home/image'
outpath = r'/home/output'

if not os.path.exists(outpath):
    os.mkdir(outpath)
for imagepath in os.listdir(filepath):
    image = io.imread(os.path.join(filepath,imagepath))
    semantic = generator_g(normalize(tf.expand_dims(image, 0)))[0]
    io.imsave(os.path.join(outpath,imagepath), semantic)

EPOCHS = 10
