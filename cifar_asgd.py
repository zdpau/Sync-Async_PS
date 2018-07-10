from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import tensorflow as tf
import cifar10
import cifar10_train
import time
from collections import deque
import random
import sys

numLoops = 2500


FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('param_name', 'default_val, 'description')
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_nodes', 1,
                            """Number of nodes.""")
tf.app.flags.DEFINE_float('delay', 0, """delay""")
tf.app.flags.DEFINE_boolean('sync', False, """synchronous mode""")
tf.app.flags.DEFINE_boolean('serial', False, """serial mode""")

def t():
    return time.time()


@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values, num_nodes):
        self.grad_buf = deque()
        values = [value.copy() for value in values]
        self.weights = dict(zip(keys, values))
        self.num_nodes = num_nodes

    def push(self, keys, values):
        # print (a)
        timeline = (t(), keys, values)
        #print(timeline)
        self.grad_buf.append(timeline)
        # print (grad_buf)

    def update(self, keys, values):
        for key, value in zip(keys, values):
            self.weights[key] += value / self.num_nodes

    def pull(self, keys):
        tau0 = t()
        while len(self.grad_buf) > 0:
            if self.grad_buf[0][0] < tau0 - FLAGS.delay:
                entry = self.grad_buf.popleft()
                self.update(entry[1], entry[2])
            else:
                break

        return [self.weights[key] for key in keys]


@ray.remote
class Worker(object):
    def __init__(self, ps, num, zero):
        self.net = cifar10_train.Train()
        self.keys = self.net.get_weights()[0]
        self.zero = zero
        self.num = num
        self.ps = ps
        self.counter = 0
        self.indexes = list(range(len(self.net.images)))
        random.shuffle(self.indexes)
        weights = ray.get(self.ps.pull.remote(self.keys))
        self.net.set_weights(self.keys, weights)        
        
    def execOne(self, c):
        index = self.indexes[c % len(self.net.images)]
        im = self.net.images[index]
        lb = self.net.labels[index]
        gradients = self.net.compute_update(im,lb)
        #2 is the node id,
        #3 is the count of batches / nodes,
        #4 is time spent from the beginning of the computation,
        #5 is the loss.
        print ("LOSS {} {} {:.6f} {}".format(self.num, c, time.time() - self.zero, self.net.lossval))
        sys.stdout.flush()
        return gradients


    def computeOneCycle(self):
        weights = ray.get(self.ps.pull.remote(self.keys))
        self.net.set_weights(self.keys, weights)

        gradients = self.execOne(self.counter)
        self.counter += 1

        self.ps.push.remote(self.keys, gradients)
        return 1 # dummy to sync

    def go(self, times, independent=False):
        for c in range(times):
            if independent:
                self.execOne(c)
            else:
                self.computeOneCycle()
        return 1

def main(argv=None):

    # tf.app.flags.FLAGS._parse_flags(sys.argv)
#    cifar10.maybe_download_and_extract()

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    ray.init(num_gpus=2)

    net = cifar10_train.Train()
    all_keys, all_values = net.get_weights()
    ps = ParameterServer.remote(all_keys, all_values, FLAGS.num_nodes)

    zero = time.time()
    workers = [Worker.remote(ps, n, zero) for n in range(FLAGS.num_nodes)]

    if FLAGS.sync:
        print("SYNC mode")
        for _ in range(numLoops):
            ray.get([w.computeOneCycle.remote() for w in workers])
    elif FLAGS.serial:      
        print("SERIAL mode")
        _ = ray.get(workers[0].go.remote(numLoops, independent=True))
    else:
        print("ASYNC mode")
        _ = ray.get([w.go.remote(numLoops, independent=False) for w in workers])

        
                

if __name__ == '__main__':
    tf.app.run()
