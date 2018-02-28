import tensorflow as tf
import dataset
from model import DCGAN

batch_size = 128
num_epoch = 70
noise_dim = 64

def main():
    train_images, test_images, train_labels, test_labels = dataset.create_mnist_paths_and_labels('./mnist_png') #dataset.train_test_data('./PetImages_resize', batch_size)
    train_size = len(train_labels)
    test_size = len(test_labels)

    noise_generator = dataset.make_noise_generator(batch_size, noise_dim)

    train_dataset = dataset.create_dataset(train_images, train_labels, img_format='png')
    train_dataset = train_dataset.cache().shuffle(buffer_size=train_size).batch(batch_size).repeat(num_epoch).make_one_shot_iterator().get_next()
    test_dataset = dataset.create_dataset(test_images, test_labels, img_format='png')
    test_dataset = test_dataset.cache().shuffle(buffer_size=10).batch(test_size).make_one_shot_iterator().get_next()


    with tf.Session() as sess:
        model = DCGAN(sess, train_dataset=train_dataset, test_dataset=test_dataset, train_size=train_size, test_size=test_size, noise_generator = noise_generator, batch_size=batch_size, num_epoch=num_epoch)
        model.build_model()
        model.intialize_variables()
        #model.create_image_from_generator()
        model.train()
        #model.test()

if __name__ == '__main__':
    main()
