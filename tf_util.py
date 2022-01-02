# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import os
import math
from pc_distance import tf_nndistance, tf_approxmatch
from tf_ops.grouping.tf_grouping import query_ball_point, group_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point

def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update


def get_repulsion_loss4(pred, nsample=20, radius=0.02):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if weight_decay>0:
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer=None
        outputs = tf.contrib.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                            activation_fn=activation_fn,weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=regularizer,
                                           biases_regularizer=regularizer)
        return outputs
        
def gen_grid_up(up_ratio):
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_ratio%i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.linspace(-0.2, 0.2, num_x)
    grid_y = tf.linspace(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def gen_grid(num_grid_point):
    """
    output [num_grid_point, 2]
    """
    x = tf.linspace(-0.05, 0.05, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def knn_sample(nsample ,xyz , new_xyz):
    _,idx = knn_point(nsample, xyz, new_xyz)
    group_xyz = group_point(xyz, idx)
    out = group_xyz[0][0]
    return out


def symmetric_sample(points, num):
    p1_idx = farthest_point_sample(num, points)
    input_fps = gather_point(points, p1_idx)
    input_fps_flip = tf.concat(
        [tf.expand_dims(input_fps[:, :, 0], axis=2), tf.expand_dims(input_fps[:, :, 1], axis=2),
         tf.expand_dims(-input_fps[:, :, 2], axis=2)], axis=2)
    input_fps = tf.concat([input_fps, input_fps_flip], 1)
    return input_fps

def contract_expand_operation(inputs,up_ratio):
    net = inputs
    net = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
    net = tf.transpose(net, [0, 2, 1, 3])

    net = conv2d(net,
                       64,
                       [1, up_ratio],
                       scope='down_conv1',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net = conv2d(net,
                       128,
                       [1, 1],
                       scope='down_conv2',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net = tf.reshape(net, [tf.shape(net)[0], -1, up_ratio,64])
    net = conv2d(net,
                       64,
                       [1, 1],
                       scope='down_conv3',
                       stride=[1, 1],
                       padding='VALID',
                       weight_decay=0.00001,
                       activation_fn=tf.nn.relu,
                       reuse=tf.AUTO_REUSE)
    net=tf.reshape(net,[tf.shape(net)[0], -1, 64])
    return net