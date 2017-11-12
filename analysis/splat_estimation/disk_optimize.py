"""
Minimize the vertex projection error onto the splats
"""
import numpy as np
import tensorflow as tf
from diffrend.model import load_obj, compute_circum_circle, compute_face_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Number of splats
num_splats = 1000

b_normalize = False

obj = load_obj('../../data/bunny.obj')

P = obj['v']
if b_normalize:
    P = (P - np.mean(P, axis=0)) / np.max(np.max(P, axis=0) - np.min(P, axis=0))

cc = compute_circum_circle(obj)
fn = compute_face_normal(obj)
rand_idx = np.random.randint(0, P.shape[0], num_splats)

cc_c = cc['center']
init_c = cc['center'][rand_idx]

C = tf.Variable(init_c, name='center')
R = tf.Variable(cc['radius'][rand_idx], name='radius')
N = tf.Variable(fn[rand_idx])
N = tf.identity(N / tf.reduce_sum(N ** 2, axis=-1)[..., tf.newaxis], name='normal')

# compute projection error
PC_diff = P[:, np.newaxis, :] - C[tf.newaxis, ...]  # vector from the splat center to the points
NPC = tf.reduce_sum(PC_diff * N[tf.newaxis, ...], axis=-1)  # projection onto the normal
NPC_N = NPC[..., tf.newaxis] * N[tf.newaxis, ...]  # projected vector along the normal
U = PC_diff - NPC_N  # vector on the splat surface

dist_within_r_th = tf.cast(tf.reduce_sum(U ** 2, axis=-1) <= (R[tf.newaxis, :] ** 2), dtype=NPC.dtype)
dist_within_r = tf.sigmoid( ((R[tf.newaxis, :] ** 2) - tf.reduce_sum(U ** 2, axis=-1)) )
#error = tf.reduce_sum(tf.abs(NPC) * dist_within_r)

# Mean position
mean_pos = np.mean(P, axis=0)
bounds = np.max(P, axis=0) - np.min(P, axis=0)

# Among the splats that a point belongs to, find the closest splat.
threshold = tf.cast(dist_within_r > .5, dtype=NPC.dtype)  # belongs to a splat
aug_dist = tf.abs(NPC) + 1 / (threshold + 1e-8)
min_indices = tf.argmin(aug_dist, axis=-1)
row_indices = np.arange(P.shape[0])
indices = tf.concat((row_indices[:, np.newaxis], tf.reshape(min_indices, (-1, 1))), axis=1)
error = tf.reduce_sum(tf.gather_nd(tf.abs(NPC), indices))

at_least_one = tf.reduce_sum(1 / (tf.reduce_sum(tf.cast(dist_within_r > .5, dtype=NPC.dtype), axis=-1) + 1e-10))
loss = error + tf.reduce_sum(1 / (tf.abs(R) * 1000)) + at_least_one
lr = 1e-3
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
max_iter = 10000
print_interval = 100
loss_per_iter = []
err_per_iter = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    C_, R_, N_, err_, dist_, loss_ = sess.run([C, R, N, error, dist_within_r, loss])
    print('Init loss: %.4f, error: %.4f' % (loss_, err_))
    for iter in range(max_iter):
        sess.run(opt)
        C_, R_, N_, err_, dist_, dist_th_, at_least_one_, loss_, U_, NPC_ = sess.run([C, R, N, error, dist_within_r, dist_within_r_th, at_least_one, loss, U, NPC])
        err_per_iter.append(err_)
        loss_per_iter.append(loss_)
        if iter % print_interval == 0 or iter == max_iter - 1:
            print('%d. Loss: %.4f, error: %.4f, at least 1 loss: %.4f' % (iter + 1, loss_, err_, at_least_one_))


plt.figure()
plt.plot(loss_per_iter)

plt.figure()
plt.scatter(C_[:, 0], C_[:, 1])
plt.scatter(init_c[:, 0], init_c[:, 1])


fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], zdir='y', s=1)
plt.title('Original vertices')


fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(init_c[:, 0], init_c[:, 1], init_c[:, 2], zdir='y', s=1)
plt.title('Initial splats')

fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(C_[:, 0], C_[:, 1], C_[:, 2], zdir='y', s=1)
plt.title('Final splats')
