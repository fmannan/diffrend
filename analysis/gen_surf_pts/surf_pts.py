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
    assert pt.shape[-1] == 3
    m, v = tf.nn.moments(pts, axes=1)
    return (scale * (radius ** 2 - tf.reduce_sum(pt ** 2, axis=-1))) ** 2 + tf.reduce_sum(tf.abs(m)) + \
           1. / (tf.reduce_sum(v) + 1e-10)


def point_on_circle_xy_loss(pt, radius, scale=1.0):
    assert pt.shape[-1] == 3
    #circle_xy = tf.constant(np.array([radius, radius, 0]), dtype=pt.dtype)[tf.newaxis, :]
    m, v = tf.nn.moments(pts, axes=1)
    return (scale * (radius ** 2 - tf.reduce_sum(pt[..., :2] ** 2, axis=-1))) ** 2 + (scale * pt[..., 2]) ** 2 + \
        tf.reduce_sum(tf.abs(m)) + 1. / (tf.reduce_sum(v) + 1e-10)


def point_on_disk_xy_loss(pt, radius, scale=1.0):
    assert pt.shape[-1] == 3

    # x^2 + y^2 <= r^2 => x^2 + y^2 - r^2 <= 0
    xy_dist_sqr = tf.reduce_sum(pt[..., :2] ** 2, axis=-1)

    m, _ = tf.nn.moments(pts, axes=1)
    _, v = tf.nn.moments(pts[..., :2], axes=1)

    return (scale * tf.maximum(0.0,  xy_dist_sqr - radius ** 2)) ** 2 + \
           (scale * pt[..., 2]) ** 2 + \
           scale / (tf.pow(xy_dist_sqr, 0.25) + 1e-10) + \
           tf.reduce_sum(tf.abs(m)) + 1. / (tf.reduce_sum(v) + 1e-10)


def point_on_cube_loss(pt, side, scale=1.0):
    pass


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


def conv_gen_0(x, is_training, **params):
    N = params['input_size']
    x = tf.reshape(x, (N, 1, 1, -1))
    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            x = slim.conv2d(x, 25, (1, 1))
            x = tf.reshape(x, (N, 5, 5, 1))
            x = slim.conv2d(x, 49, (3, 3), padding='VALID')
            x = slim.fully_connected(x, params['output_size'], scope='out', activation_fn=None, normalizer_fn=None)
    return x


def conv_gen_1(x, is_training, **params):
    N = params['input_size']
    x = tf.reshape(x, (N, 1, 1, -1))
    with tf.variable_scope('generator'):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            x = slim.conv2d(x, 25, (1, 1))
            x = tf.reshape(x, (N, 5, 5, 1))
            x = slim.conv2d(x, 49, (3, 3), padding='SAME')
            x = tf.reshape(x, (N, 7, 7, -1))
            x = slim.conv2d(x, 81, (3, 3), padding='VALID')
            x = slim.fully_connected(x, params['output_size'], scope='out', activation_fn=None, normalizer_fn=None)
    return x


if __name__ == '__main__':
    input_size = 1000  # batch of inputs
    rand_noise_size = 100  # dimension of the input space
    num_points = 1000
    output_size = num_points * 3  # generate 1000 points

    lr = 1e-4

    net_0_spec = [4096, 100]
    net_1_spec = [4096, 100, 4096]
    net_2_spec = [1024, 8192, 4096, 100]

    net_spec = net_2_spec
    fc_net_fn = lambda x: fc_net(x, layers=net_spec, is_training=True, output_size=output_size)
    #conv_net_fn = lambda x: conv_gen_0(x, is_training=True, input_size=input_size, output_size=output_size)
    conv_net_fn = lambda x: conv_gen_1(x, is_training=True, input_size=input_size, output_size=output_size)
    network_fn = fc_net_fn  #conv_net_fn  #

    obj_fn = point_on_sphere_loss  #point_on_circle_xy_loss  #point_on_disk_xy_loss #

    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(dtype=tf.float32, shape=[input_size, rand_noise_size])

        pts = network_fn(X)
        print(input_size, output_size)
        pts = tf.reshape(pts, (input_size, num_points, 3))
        print(pts)

        loss = tf.reduce_mean(obj_fn(pts, radius=1., scale=10))

        opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    max_iter = 7000
    print_interval = 100
    loss_per_iter = []
    err_per_iter = []
    best_loss = np.inf
    best_config = dict()

    IMG_DIR = './sphere_res_conv_gen_1_v0_1' #'./circle_res_conv_2'
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    fig0 = plt.figure(figsize=(9, 3))
    ax0 = fig0.add_subplot(131)
    ax1 = fig0.add_subplot(132)
    ax2 = fig0.add_subplot(133)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        for idx in range(max_iter):
            rand_input = random_sample_generator([input_size, rand_noise_size])
            _, loss_ = sess.run([opt, loss], feed_dict={X: rand_input})

            if idx % print_interval == 0 or idx == max_iter - 1:
                print('%d. Loss: %.4f' % (idx, loss_))
                pts_ = sess.run(pts, feed_dict={X: rand_input})
                ax.clear()
                ax.scatter(pts_[0, :, 0], pts_[0, :, 1], pts_[0, :, 2], s=1)
                ax.view_init(20, idx % 360)
                ax.set_aspect('equal')
                plt.title('Generated output {}, loss: {:.4f}'.format(idx, loss_))
                plt.xlabel('x')
                plt.ylabel('y')
                fig.savefig(IMG_DIR + '/fig_{:06d}.png'.format(idx))

                # 2D projections
                ax0.clear()
                ax0.scatter(pts_[0, :, 0], pts_[0, :, 1], s=1)
                ax0.set_xlim(-1, 1)
                ax0.set_ylim(-1, 1)
                ax0.title.set_text('XY')
                ax0.set_aspect('equal')

                ax1.clear()
                ax1.scatter(pts_[0, :, 1], pts_[0, :, 2], s=1)
                ax1.set_xlim(-1, 1)
                ax1.set_ylim(-1, 1)
                ax1.title.set_text('YZ')
                ax1.set_aspect('equal')

                ax2.clear()
                ax2.scatter(pts_[0, :, 0], pts_[0, :, 2], s=1)
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.title.set_text('XZ')
                ax2.set_aspect('equal')
                fig0.savefig(IMG_DIR + '/fig_proj_{:06d}.png'.format(idx))


