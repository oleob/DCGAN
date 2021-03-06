import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm
import cv2

class DCGAN(object):

    def __init__(self, sess, train_dataset, test_dataset, train_size, test_size, batch_size, num_epoch):
        self.sess = sess
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_epoch = num_epoch

    def conv2d(self, inp, num_filters, filter_size, strides, padding='valid', name='conv2d'):
        conv = tf.layers.conv2d(inp, num_filters, filter_size, strides = strides, padding=padding, name='name')
        #conv = tf.layers.batch_normalization()

    def build_model(self):
        #Initialize discriminator layers
        self.d_input_shape = [-1, 64, 64, 3] #TODO: change to numpy array
        self.d_inputs = tf.placeholder(tf.float32, [None, 64*64*3], name='input_img')
        self.labels = tf.placeholder(tf.int32, [None], name='input_label')
        self.conv1 = tf.layers.Conv2D(16, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu, name = 'd_conv1') #output_dim = [31,31]
        self.conv1_batch_norm = tf.layers.BatchNormalization(trainable=True, name='d_conv1_batch_norm')
        self.conv2 = tf.layers.Conv2D(32, 5, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu, name = 'd_conv2') #output_dim = [14,14]
        self.conv2_batch_norm = tf.layers.BatchNormalization(trainable=True, name='d_conv1_batch_norm')
        self.conv3 = tf.layers.Conv2D(32, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu, name = 'd_conv3') #output_dim = [6,6]
        self.conv3_batch_norm = tf.layers.BatchNormalization(trainable=True, name='d_conv1_batch_norm')
        self.conv4 = tf.layers.Conv2D(32, 4, strides=[2,2], padding='valid', activation= tf.nn.leaky_relu, name = 'd_conv4') #output_dim = [2,2]
        self.conv4_batch_norm = tf.layers.BatchNormalization(trainable=True, name='d_conv1_batch_norm')
        self.fc1 = tf.layers.Dense(32*2*2, activation=tf.nn.leaky_relu, name='d_dense1')
        self.fc2 = tf.layers.Dense(1)
        #Initialize generator layers

        self.g_input_shape = [-1,10,10,3] #TODO: change to numpy array
        self.g_inputs = tf.placeholder(tf.float32, [None, 10*10*3], name='input_noise')
        self.deconv1 = tf.layers.Conv2DTranspose(32, 4, strides=[2,2], padding='valid', activation=tf.nn.relu, name = 'g_deconv1') #output_dim = [22,22]
        self.deconv2 = tf.layers.Conv2DTranspose(32, 4, strides=[2,2], padding='valid', activation=tf.nn.relu, name = 'g_deconv2') #output_dim = [46,46]
        self.deconv3 = tf.layers.Conv2DTranspose(32, 8, strides=[1,1], padding='valid', activation=tf.nn.relu, name = 'g_deconv3') #output_dim = [53,53]
        self.deconv4 = tf.layers.Conv2DTranspose(32, 8, strides=[1,1], padding='valid', activation=tf.nn.relu, name = 'g_deconv4') #output_dim = [60,60]
        self.deconv5 = tf.layers.Conv2DTranspose(3, 5, strides=[1,1], padding='valid', activation=tf.nn.relu, name = 'g_deconv5') #output_dim = [64,64]

        #Initialize Discriminator and Generator
        self.G = self.generator()
        self.D_real, self.D_real_logits = self.discriminator(self.d_inputs)
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True)

        #Define loss and optimizers
        self.D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real_logits)))
        self.D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake_logits)))
        self.D_loss = self.D_real_loss + self.D_fake_loss
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake_logits)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.D_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.D_loss, var_list=self.d_vars)
        self.G_optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.G_loss, var_list=self.g_vars)

        #set up merge
        self.merge = tf.summary.merge_all() #TODO use this instead of local defined merge
        #add scalars for tensorboard
        tf.summary.scalar('Discriminator loss', self.D_loss)
        tf.summary.scalar('Generator loss', self.G_loss)

    def discriminator(self,inputs, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            d = tf.reshape(inputs, self.d_input_shape)
            d = self.conv1(d)
            d = self.conv1_batch_norm(d)
            d = self.conv2(d)
            d = self.conv2_batch_norm(d)
            d = self.conv3(d)
            d = self.conv3_batch_norm(d)
            d = self.conv4(d)
            d = self.conv4_batch_norm(d)
            d = tf.layers.flatten(d)
            d = self.fc1(d)
            d = self.fc2(d)
        return tf.nn.sigmoid(d), d

    def generator(self):
        g = tf.reshape(self.g_inputs, self.g_input_shape)
        g = self.deconv1(g)
        g = self.deconv2(g)
        g = self.deconv3(g)
        g = self.deconv4(g)
        g = self.deconv5(g)
        g = tf.reshape(g,[-1, 64*64*3])
        g = tf.nn.tanh(g)
        return g

    def intialize_variables(self):
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def train(self):

        summary_writer = tf.summary.FileWriter( './logs/1/train', self.sess.graph)
        num_batches_in_epoch = math.ceil(self.train_size/self.batch_size)
        merge = tf.summary.merge_all()
        for i in  tqdm(range(num_batches_in_epoch*self.num_epoch)):
            imgs, labels = self.sess.run(self.train_dataset)
            noise = np.random.uniform(-1.0, 1.0, (self.batch_size, 100*3)).astype(np.float32)

            summary = self.sess.run([self.D_optimize, self.G_optimize, merge], feed_dict={self.d_inputs: imgs, self.labels: labels, self.g_inputs: noise})
            #self.sess.run(self.G_optimize, feed_dict={self.d_inputs: imgs, self.labels: labels, self.g_inputs: noise})
            summary_writer.add_summary(summary[2],i)
            if (i)%num_batches_in_epoch == 0:
                self.sample(i)

    def test(self):
        img, label = self.sess.run(self.test_dataset)
        acc = self.sess.run(self.D_accuracy, feed_dict={self.d_inputs: img, self.labels: label})
        print('Test accuracy:', acc[1])

    def create_image_from_generator(self):
        noise = np.random.uniform(0, 1, (self.batch_size, 100*3)).astype(np.float32)
        generated_imgs = self.sess.run(self.G, feed_dict={self.g_inputs: noise})
        generated_imgs = tf.reshape(generated_imgs,[self.batch_size, 64*64*3])
        labels = [0]*self.batch_size
        return generated_imgs, labels
        #plt.imshow(generated_imgs[0], interpolation='nearest')
        #plt.show()
    def sample(self, iteration):
        noise = np.random.uniform(0, 1, (1, 100*3)).astype(np.float32)
        generated_imgs = self.sess.run(self.G, feed_dict={self.g_inputs: noise})
        generated_imgs = np.reshape(generated_imgs,[1, 64, 64, 3])
        img = (255.0*(generated_imgs[0]+1.0))/2.0
        cv2.imwrite('generator_images/filename_'+ str(iteration) + '.png', img)
