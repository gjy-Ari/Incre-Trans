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
      image, size=[1024, 1024, 3])
  return cropped_image

def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  image = tf.image.resize(image, [1054, 1054],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  image = random_crop(image)
  image = tf.image.random_flip_left_right(image)
  return image

def preprocess_image_train(image):
    image_string = tf.read_file(image                   )
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


paths1 = walk_type(r'','*.png') # image path1
filenames4 = tf.constant(paths1)
import random
paths4 = []
for i in range(2000):
  paths4.append(random.sample(paths1,1)[0])
  
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

# Save model
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


EPOCHS = 10


@tf.function
def train_step(real_x, real_y):
    # The GradientTape is applied multiple times to calculate the gradient.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G converts X -> Y.
        # Generator F converts Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y is used for loss of consistency.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator losses
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Loss of generators and discriminators.
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))

    return gen_g_loss,total_cycle_loss,identity_loss(real_y, same_y)

Gen_g_loss = []
Total_cycle_loss = []
Identity_loss = []
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
    b = train_step(image_x, image_y)
    Gen_g_loss.append(float(b[0]))
    Total_cycle_loss.append(float(b[1]))
    Identity_loss.append(float(b[2]))
    if n % 10 == 0:
      print (epoch,'|',n)
      print(float(b[0]),float(b[1]),float(b[2]))
    n+=1

  #clear_output(wait=True)


  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

import numpy as np
np.save('Gen_g_loss.npy',np.array(Gen_g_loss))
np.save('Total_cycle_loss.npy',np.array(Total_cycle_loss))
np.save('Identity_loss.npy',np.array(Identity_loss))
