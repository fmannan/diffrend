"""
take a random vector and generate points on a sphere
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def point_on_sphere_loss(pt, radius, scale=1.0):
    assert pt.shape[1] == 3
    return (scale * (radius ** 2 - tf.reduce_sum(pt ** 2, axis=-1))) ** 2


def point_on_circle_xy_loss(pt, radius, scale=1.0):
    assert pt.shape[1] == 3
    #circle_xy = tf.constant(np.array([radius, radius, 0]), dtype=pt.dtype)[tf.newaxis, :]
    return (scale * (radius ** 2 - tf.reduce_sum(pt[:, :2] ** 2, axis=-1))) ** 2 + (scale * pt[:, 2]) ** 2


def random_sample_generator(shape):
    return np.random.uniform(-1., 1., size=shape)


def net_0(x, is_training, **params):
    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.fully_connected(x, 4096, scope='fc1', normalizer_fn=slim.batch_norm)
            net = slim.fully_connected(net, 100, scope='fc2')
            net = slim.fully_connected(net, params['output_size'], scope='out', activation_fn=None, normalizer_fn=None)

    return net


def fc_net(x, is_training, **params):
    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = x
            for layer_id, sz in enumerate(params['layers']):
                net = slim.fully_connected(net, sz, scope='fc_{}'.format(layer_id), normalizer_fn=slim.batch_norm)
            net = slim.fully_connected(net, params['output_size'], scope='out', activation_fn=None, normalizer_fn=None)

    return net

input_size = 1000
rand_noise_size = 2
lr = 1e-4

net_0_spec = [4096, 100]
net_1_spec = [4096, 100, 4096]
net_2_spec = [1024, 8192, 4096, 100]

obj_fn = point_on_sphere_loss  #point_on_circle_xy

graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[input_size, rand_noise_size])
    pts = fc_net(X, layers=net_2_spec, is_training=True, output_size=3)

    loss = tf.reduce_mean(obj_fn(pts, radius=1., scale=10))

    opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

max_iter = 10000
print_interval = 100
loss_per_iter = []
err_per_iter = []
best_loss = np.inf
best_config = dict()

IMG_DIR = './sphere_out'
if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    for idx in range(max_iter):
        rand_input = random_sample_generator([input_size, rand_noise_size])
        _, loss_ = sess.run([opt, loss], feed_dict={X: rand_input})

        if idx % print_interval == 0:
            print('%d. Loss: %.4f' % (idx, loss_))
            pts_ = sess.run(pts, feed_dict={X: rand_input})
            ax.clear()
            ax.scatter(pts_[:, 0], pts_[:, 1], pts_[:, 2], s=1)
            plt.title('Generated output {}'.format(idx))
            fig.savefig(IMG_DIR + '/fig_{:06d}.png'.format(idx))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pts_[:, 0], pts_[:, 1], pts_[:, 2], s=1.2)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# rand_points = np.random.rand(1000, 3)
# ax.scatter(rand_points[:, 0], rand_points[:, 1], rand_points[:, 2])
#

#
# ax.clear()
# ax.scatter(pts_[:, 0], pts_[:, 1], pts_[:, 2], s=1)
# plt.title('Generated output {}'.format(idx))
# fig.savefig(IMG_DIR + '/fig_{:06d}.png'.format(idx))