import tensorflow as tf
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np

def create_mnist_paths_and_labels(directory):
    train_path = directory + '/training'
    test_path = directory + '/testing'
    train_paths, train_labels = create_paths_and_labels(train_path)
    test_paths, test_labels = create_paths_and_labels(test_path)
    return (train_paths, test_paths, train_labels, test_labels)

def create_lfw_paths_and_labels(directory):
    directory = './lfw'
    image_paths = []
    image_labels = []
    #maleNames = open('male_names.txt','r').read().split('\n')

    for folder in listdir(directory):
        folder_path = join(directory,folder)
        for image in listdir(folder_path):
            image_path = join(folder_path,image)
            image_paths.append(image_path)
            image_labels.append(1)
            # if image in maleNames:
            #     image_labels.append([1])
            # else:
            #     image_labels.append([0])
    return image_paths, image_labels

def create_paths_and_labels(directory, GAN=False):
    image_paths = []
    image_labels = []
    folders = listdir(directory)
    for i in range(len(folders)):
        folder = folders[i]
        folder_path = join(directory,folder)
        for image in listdir(folder_path):
            image_path = join(folder_path,image)
            image_paths.append(image_path)
            if GAN:
                image_labels.append(1)
            else:
                image_labels.append([i])
    return image_paths, image_labels

def create_dataset(image_paths, image_labels, img_format='jpg'):
    def decode_image(image_path):
        image_string = tf.read_file(image_path)
        if img_format=='jpg':
            image = tf.image.decode_jpeg(image_string)
        elif img_format=='png':
            image = tf.image.decode_png(image_string)
        else:
            image = tf.image.decode(image_string)
        image = tf.image.resize_images(image, [28, 28])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image,[28*28*1])
        return 1.0 - (2.0*image)/255.0
    images = tf.data.Dataset.from_tensor_slices(image_paths).map(decode_image)
    labels = tf.data.Dataset.from_tensor_slices(image_labels)
    return tf.data.Dataset.zip((images, labels))

def train_test_data(directory, batch_size):
    image_paths, image_labels = create_lfw_paths_and_labels(directory) #TODO replace with ordinary function
    #image_paths = image_paths[len(image_paths)%batch_size-1:]
    #image_labels = image_labels[len(image_labels)%batch_size-1:]

    train_images, test_images, train_labels, test_labels = train_test_split(image_paths, image_labels, stratify=image_labels, test_size=0.25)
    return (train_images, test_images, train_labels, test_labels)

def make_noise_generator(batch_size, noise_dim):
    def gen():
        while True:
            yield np.random.uniform(-1.0, 1.0, (batch_size, 8*8*1)).astype(np.float32) #TODO replace with tf.random_normal
    generator = tf.data.Dataset.from_generator(gen, output_types=tf.float32, output_shapes=[batch_size, noise_dim])
    return generator.make_one_shot_iterator().get_next()

# def dataset(directory):
#
#     image_paths = []
#     image_labels = []
#     folders = listdir(directory)
#     for i in range(len(folders)):
#         folder = folders[i]
#         print(folder)
#         folder_path = join(directory,folder)
#         for image in listdir(folder_path):
#             image_path = join(folder_path,image)
#             image_paths.append(image_path)
#             #image_labels.append(folder)
#             image_labels.append([i])
#     def decode_image(image_path):
#         image_string = tf.read_file(image_path)
#         image = tf.image.decode_jpeg(image_string)
#         image = tf.image.resize_images(image, [64, 64])
#         image = tf.cast(image, tf.float32)
#         image = tf.reshape(image,[64*64*3])
#         return image/255.0
#     #images = tf.data.FixedLengthRecordDataset(image_paths,64*64).map(decode_image)
#     #labels = tf.data.FixedLengthRecordDataset(image_labels,1)
#     images = tf.data.Dataset.from_tensor_slices(image_paths).map(decode_image)
#     labels = tf.data.Dataset.from_tensor_slices(image_labels)
#     return tf.data.Dataset.zip((images, labels))
#
# def train(directory):
#     image_paths, image_labels = create_paths_and_labels(directory)
#     return create_dataset(image_paths, image_labels)
#
# def test(directory):
#     return dataset(directory)
