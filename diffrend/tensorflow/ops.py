import tensorflow as tf


def nonzero_divide(x, y):
    denom = tf.where(tf.abs(y) > 0, y, tf.ones_like(y))
    return x / denom


def norm(u, p):
    return tf.reduce_sum(u ** p, axis=-1)


def normalize(u):
    denom = norm(u, 2)
    return nonzero_divide(u, denom[..., tf.newaxis])
