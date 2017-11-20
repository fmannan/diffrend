"""
Minimize the vertex projection error onto the splats
"""
import numpy as np
import tensorflow as tf
from diffrend.model import load_obj, load_off, compute_circum_circle, compute_face_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Number of splats
num_splats = 1000

b_normalize = True

#obj = load_obj('../../data/bunny.obj')
obj = load_off('../../data/chair_0001.off')

P = obj['v']
if b_normalize:
    P = (P - np.mean(P, axis=0)) / np.max(np.max(P, axis=0) - np.min(P, axis=0))
obj['v'] = P

cc = compute_circum_circle(obj)
fn = compute_face_normal(obj)

cc_c = cc['center']
P = cc_c
PN = fn
rand_idx = np.random.randint(0, P.shape[0], num_splats)
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

# Normal alignment loss 1 - dot(vertex_normal, splat_normal)
PN_N_dot = tf.matmul(fn, tf.transpose(N, (1, 0)))
PN_N_error = 1 - PN_N_dot

# Among the splats that a point belongs to, find the closest splat.
threshold = tf.cast(dist_within_r > .5, dtype=NPC.dtype)  # belongs to a splat
aug_dist = tf.abs(NPC) + 1 / (threshold + 1e-8) + PN_N_error * 100
min_indices = tf.argmin(aug_dist, axis=-1)
row_indices = np.arange(P.shape[0])
indices = tf.concat((row_indices[:, np.newaxis], tf.reshape(min_indices, (-1, 1))), axis=1)

# select the splats that each face/vertex belongs to
selected_splat_dist = tf.gather_nd(tf.abs(NPC), indices)
selected_splat_orien = tf.gather_nd(tf.abs(PN_N_error), indices)

splat_to_point_dist = tf.reduce_sum(selected_splat_dist)
splat_to_face_orientation = tf.reduce_sum(selected_splat_orien)
error = splat_to_point_dist + splat_to_face_orientation

# Bounded splat radius loss
bounded_radius_loss = tf.reduce_mean(R) * 100

# Not too much more than 1
at_most_few = tf.reduce_mean(tf.reduce_sum(tf.cast(dist_within_r > .5, dtype=NPC.dtype), axis=-1))

at_least_one = tf.reduce_sum(1 / (tf.reduce_sum(tf.cast(dist_within_r > .5, dtype=NPC.dtype), axis=-1) + 1e-10))
nonzero_radius_loss = tf.reduce_sum(1 / (tf.abs(R) * 1000))
loss = error + nonzero_radius_loss + bounded_radius_loss + at_least_one + at_most_few
lr = 1e-4
opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
max_iter = 4000
print_interval = 100
loss_per_iter = []
err_per_iter = []
best_loss = np.inf
best_config = dict()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    C_, R_, N_, err_, dist_, loss_ = sess.run([C, R, N, error, dist_within_r, loss])
    print(loss_, err_)
    print('Init loss: %.4f, error: %.4f' % (loss_, err_))
    for iter in range(max_iter):
        sess.run(opt)
        C_, R_, N_, err_, dist_, dist_th_, at_least_one_, loss_, U_, NPC_, PN_N_dot_, \
        splat_face_dist_, splat_face_theta_, selected_splat_dist_, selected_splat_orien_ = sess.run([C, R, N, error, dist_within_r, dist_within_r_th,
                                                        at_least_one, loss, U, NPC, PN_N_dot,
                                                        splat_to_point_dist, splat_to_face_orientation,
                                                        selected_splat_dist, selected_splat_orien])
        err_per_iter.append(err_)
        loss_per_iter.append(loss_)
        if loss_ < best_loss:
            best_config['C'] = C_
            best_config['R'] = R_
            best_config['N'] = N_
            best_config['splat_dist'] = selected_splat_dist_
            best_config['splat_orientation_error'] = selected_splat_orien_
            best_config['splat_dist_sum'] = splat_face_dist_
            best_config['splat_orientation_sum'] = splat_face_theta_
            best_loss = loss_

        if iter % print_interval == 0 or iter == max_iter - 1:
            print('%d. Loss: %.4f, error: %.4f, dist: %.4f, orientation: %.4f, at least 1 loss: %.4f, cbest: %.4f' %
                  (iter + 1, loss_, err_, splat_face_dist_, splat_face_theta_, at_least_one_, best_loss))

num_vertices = best_config['splat_orientation_error'].shape[0]
num_of_misoriented = np.sum(best_config['splat_orientation_error'] > 1.)
pct_misoriented = num_of_misoriented / num_vertices * 100
print('misoriented: %d, pct: %.4f%%' % (num_of_misoriented, pct_misoriented))

plt.figure()
plt.plot(loss_per_iter)

C_ = best_config['C']

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

plt.figure()
plt.hist(best_config['splat_orientation_error'])
