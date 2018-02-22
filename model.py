import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
import random

class DCGAN(object):

    def __init__(self, sess, train_dataset, test_dataset, train_size, test_size, batch_size, num_epoch):
        self.sess = sess
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def build_model(self):
        #Initialize discriminator layers
        self.d_input_shape = [-1, 64, 64, 3] #TODO: change to numpy array
        self.d_inputs = tf.placeholder(tf.float32, [None, 64*64*3], name='input_img')
        self.labels = tf.placeholder(tf.int32, [None, 1], name='input_label')
        self.inverse_labels = tf.placeholder(tf.int32, [None, 1], name='input_label')
        self.conv1 = tf.layers.Conv2D(16, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu) #output_dim = [31,31]
        self.conv2 = tf.layers.Conv2D(32, 5, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu) #output_dim = [14,14]
        self.conv3 = tf.layers.Conv2D(32, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu) #output_dim = [6,6]
        self.conv4 = tf.layers.Conv2D(32, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu) #output_dim = [2,2]
        self.fc1 = tf.layers.Dense(32*2*2, activation=tf.nn.leaky_relu)
        self.fc2 = tf.layers.Dense(2)
        #Initialize generator layers

        self.g_input_shape = [-1,10,10,3] #TODO: change to numpy array
        self.g_inputs = tf.placeholder(tf.float32, [None, 10*10*3])
        self.deconv1 = tf.layers.Conv2DTranspose(32, 4, strides=[2,2], padding='valid', activation=tf.nn.relu) #output_dim = [22,22]
        self.deconv2 = tf.layers.Conv2DTranspose(32, 4, strides=[2,2], padding='valid', activation=tf.nn.relu) #output_dim = [46,46]
        self.deconv3 = tf.layers.Conv2DTranspose(32, 8, strides=[1,1], padding='valid', activation=tf.nn.relu) #output_dim = [53,53]
        self.deconv4 = tf.layers.Conv2DTranspose(32, 8, strides=[1,1], padding='valid', activation=tf.nn.relu) #output_dim = [60,60]
        self.deconv5 = tf.layers.Conv2DTranspose(3, 5, strides=[1,1], padding='valid', activation=tf.nn.relu) #output_dim = [64,64]

        #Initialize discriminator with loss, accuracy and optimizer
        self.D_real = self.discriminator()
        self.D_fake = self.discriminator(reuse=True)
        self.D_real_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.D)
        self.D_fake_loss =
        self.D_accuracy = tf.metrics.accuracy(labels=self.labels, predictions=tf.argmax(self.D, axis=1))
        self.D_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.D_loss)
        #add scalars for tensorboard
        tf.summary.scalar('Discriminator accuracy', self.D_accuracy[1])
        tf.summary.scalar('Discriminator loss', self.D_loss)

        #Initialize generator
        self.G = self.generator()
        self.G_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.inverse_labels, logits = self.G)
        self.G_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.G_loss)
        #set up merge
        self.merge = tf.summary.merge_all() #TODO use this instead of local defined merge

    def discriminator(self, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            d = tf.reshape(self.d_inputs, self.d_input_shape)
            d = self.conv1(d)
            d = self.conv2(d)
            d = self.conv3(d)
            d = self.conv4(d)
            d = tf.layers.flatten(d)
            d = self.fc1(d)
            d = self.fc2(d)
        return d

    def generator(self):
        g = tf.reshape(self.g_inputs, self.g_input_shape)
        g = self.deconv1(g)
        g = self.deconv2(g)
        g = self.deconv3(g)
        g = self.deconv4(g)
        g = self.deconv5(g)
        return g

    def intialize_variables(self):
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def train(self):
        def invert_labels(labels):
            flat_labels = np.reshape(labels, [len(labels)])
            inverse_labels = 1-flat_labels
            inverse_labels = np.reshape(inverse_labels, [len(labels), 1])
            return inverse_labels

        summary_writer = tf.summary.FileWriter( './logs/1/train', self.sess.graph)
        num_batches_in_epoch = math.ceil(self.train_size/self.batch_size)
        merge = tf.summary.merge_all()
        for i in  tqdm(range(num_batches_in_epoch*self.num_epoch*2)):
            r = random.uniform(0.0,1.0)
            if r > 0.5:
                imgs, labels = self.sess.run(self.train_dataset)
            else:
                imgs, labels = self.create_image_from_generator()

            self.sess.run([self.D_optimize, self.G_optimize], feed_dict={self.d_inputs: imgs, self.labels: labels, self.inverse_labels: invert_labels(labels)})

            if (i/2)%num_batches_in_epoch == 0:
                acc, summary = self.sess.run([self.D_accuracy, merge], feed_dict={self.d_inputs: imgs, self.labels: labels})
                summary_writer.add_summary(summary, int(i/num_batches_in_epoch))
                self.sample()

    def test(self):
        img, label = self.sess.run(self.test_dataset)
        acc = self.sess.run(self.D_accuracy, feed_dict={self.d_inputs: img, self.labels: label})
        print('Test accuracy:', acc[1])

    def create_image_from_generator(self):
        noise = np.random.uniform(0, 1, (self.batch_size, 100*3)).astype(np.float32)
        generated_imgs = self.sess.run(self.G, feed_dict={self.g_inputs: noise})
        generated_imgs = tf.reshape(generated_imgs,[self.batch_size, 64*64*3])
        labels = [[0]]*self.batch_size
        return generated_imgs, labels
        #plt.imshow(generated_imgs[0], interpolation='nearest')
        #plt.show()
    def sample(self):
        noise = np.random.uniform(0, 1, (1, 100*3)).astype(np.float32)
        generated_imgs = self.sess.run(self.G, feed_dict={self.g_inputs: noise})
        plt.imshow(generated_imgs[0], interpolation='nearest')
        plt.show()
