from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ray
import tensorflow as tf
import cifar10
import numpy as np


numNodes = 2
numLoop = 100


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def gen_ph(variable, name=None):
    if type(variable) == np.ndarray:
        ph_type = variable.dtype
    else:
        ph_type = variable.dtype.base_dtype
    return tf.placeholder(ph_type, variable.shape, name=name)


pwd = os.getcwd()


class Train(object):
    def __init__(self):
        global pwd
        os.chdir(pwd)
        train_data = np.load('trainingdata.npz')
        self.images = train_data['images']
        self.labels = train_data['labels']
        self.counter = 0
        self.session = None

        with tf.Graph().as_default():
            global_step = tf.contrib.framework.get_or_create_global_step()

            self.images_ph = gen_ph(self.images[0], name='images_ph')
            self.labels_ph = gen_ph(self.labels[0], name='labels_ph')

            logits = cifar10.inference_divided(self.images_ph)
            self.loss = cifar10.loss(logits, self.labels_ph)

            num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                     FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

            lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                            global_step,
                                            decay_steps,
                                            cifar10.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)
#            opt = tf.train.MomentumOptimizer(lr, 0.9)
            grads_pair_list = opt.compute_gradients(self.loss)
#            self.grads = [i[0] for i in grads_pair_list]

#            self.phs = [gen_ph(i[1]) for i in grads_pair_list]
#            variables = [i[1] for i in grads_pair_list]

#            self.train_op = opt.apply_gradients(zip(self.phs, variables), global_step=global_step)
            self.train_op = opt.apply_gradients(grads_pair_list, global_step=global_step)

            self.mon_sess = tf.Session(
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement))
            tf.global_variables_initializer().run(session=self.mon_sess)
            tf.summary.FileWriter("tb", self.mon_sess.graph)

            self.variable = ray.experimental.TensorFlowVariables(self.loss, self.mon_sess)

    def compute_update(self,x,y):
        weights = self.get_weights()[1]
#        grads, self.lossval = self.mon_sess.run([self.grads, self.loss], feed_dict={self.images_ph: x, self.labels_ph:y})
#        self.mon_sess.run(self.train_op, feed_dict=dict(zip(self.phs, grads)))
        self.lossval, _= self.mon_sess.run([self.loss, self.train_op], feed_dict={self.images_ph: x, self.labels_ph:y})
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def set_weights(self, variable_names, weights):
        self.variable.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variable.get_weights()
        return list(weights.keys()), list(weights.values())
