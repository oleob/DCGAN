import tensorflow as tf
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split

def create_lfw_paths_and_labels(directory):
    directory = '../lfw'
    image_paths = []
    image_labels = []
    #maleNames = open('male_names.txt','r').read().split('\n')

    for folder in listdir(directory):
        folder_path = join(directory,folder)
        for image in listdir(folder_path):
            image_path = join(folder_path,image)
            image_paths.append(image_path)
            image_labels.append([1])
            # if image in maleNames:
            #     image_labels.append([1])
            # else:
            #     image_labels.append([0])
    return image_paths, image_labels

def create_paths_and_labels(directory):
    image_paths = []
    image_labels = []
    folders = listdir(directory)
    for i in range(len(folders)):
        folder = folders[i]
        folder_path = join(directory,folder)
        for image in listdir(folder_path):
            image_path = join(folder_path,image)
            image_paths.append(image_path)
            #image_labels.append(folder)
            image_labels.append([1])
            # if folder=='Dog':
            #     image_labels.append([1])
            # else:
            #     image_labels.append([0])
    return image_paths, image_labels

def create_dataset(image_paths, image_labels):
    def decode_image(image_path):
        image_string = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.resize_images(image, [64, 64])
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image,[64*64*3])
        return image/255.0
    images = tf.data.Dataset.from_tensor_slices(image_paths).map(decode_image)
    labels = tf.data.Dataset.from_tensor_slices(image_labels)
    return tf.data.Dataset.zip((images, labels))

def train_test_data(directory):
    image_paths, image_labels = create_lfw_paths_and_labels(directory) #TODO replace with ordinary function
    train_images, test_images, train_labels, test_labels = train_test_split(image_paths, image_labels, stratify=image_labels, test_size=0.25)
    return (train_images, test_images, train_labels, test_labels)

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
