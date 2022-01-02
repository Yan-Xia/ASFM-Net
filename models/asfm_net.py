import tensorflow as tf
import logging
from tf_util import *
import math

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s:%(name)s:%(message)s', filemode='w')

num_coarse = 1024
grid_size = 2
grid_scale = 0.05
num_fine = grid_size ** 2 * num_coarse

def create_ae1_encoder(inputs, npts):
    with tf.variable_scope('AE1_encoder_0', reuse=tf.AUTO_REUSE):
        inputs = tf.reshape(inputs, [1, -1, 3])
        features = mlp_conv(inputs, [128, 256])
        features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
        features = tf.concat([features, features_global], axis=2)
    with tf.variable_scope('AE1_encoder_1', reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, 1024])
        features = point_maxpool(features, npts)
    return features

def create_ae1_decoder(features):
    with tf.variable_scope('AE1_decoder', reuse=tf.AUTO_REUSE):
        coarse = mlp(features, [1024, 1024, num_coarse * 3])
        coarse = tf.reshape(coarse, [-1, num_coarse, 3])

    with tf.variable_scope('AE1_folding', reuse=tf.AUTO_REUSE):
        x = tf.linspace(-grid_scale, grid_scale, grid_size)
        y = tf.linspace(-grid_scale, grid_scale, grid_size)
        grid = tf.meshgrid(x, y)
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [features.shape[0], num_coarse, 1])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [-1, num_fine, 3])

        global_feat = tf.tile(tf.expand_dims(features, 1), [1, num_fine, 1])

        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
        center = tf.reshape(center, [-1, num_fine, 3])

        fine = mlp_conv(feat, [512, 512, 3]) + center
    return fine

def create_ae2_encoder(inputs, npts):
    with tf.variable_scope('AE2_encoder_0', reuse=tf.AUTO_REUSE):
        inputs = tf.reshape(inputs, [1, -1, 3])
        features = mlp_conv(inputs, [128, 256])
        features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
        logging.debug("features:{}, features_global:{}".format(features, features_global))
        features = tf.concat([features, features_global], axis=2)
    with tf.variable_scope('AE2_encoder_1', reuse=tf.AUTO_REUSE):
        features = mlp_conv(features, [512, 1024])
        features = point_maxpool(features, npts)
    return features

def create_ae2_decoder(features):
    with tf.variable_scope('AE2_decoder', reuse=tf.AUTO_REUSE):
        coarse = mlp(features, [1024, 1024, num_coarse * 3])
        coarse = tf.reshape(coarse, [-1, num_coarse, 3])

    with tf.variable_scope('AE2_folding', reuse=tf.AUTO_REUSE):
        x = tf.linspace(-grid_scale, grid_scale, grid_size)
        y = tf.linspace(-grid_scale, grid_scale, grid_size)
        grid = tf.meshgrid(x, y)
        grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
        grid_feat = tf.tile(grid, [features.shape[0], num_coarse, 1])

        point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
        point_feat = tf.reshape(point_feat, [-1, num_fine, 3])

        global_feat = tf.tile(tf.expand_dims(features, 1), [1, num_fine, 1])

        feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

        center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, grid_size ** 2, 1])
        center = tf.reshape(center, [-1, num_fine, 3])

        fine = mlp_conv(feat, [512, 512, 3]) + center
    return fine


def create_refiner(code,inputs,pre_out,step_ratio,num_extract=512):
    with tf.variable_scope('RefineUnit', reuse=tf.AUTO_REUSE):
        coarse_fps = gather_point(pre_out, farthest_point_sample(512, pre_out))
        input_fps = symmetric_sample(inputs, int(num_extract / 2))
        synthetic = tf.concat([input_fps, coarse_fps], 1)
        logging.debug('synthetic: {}'.format(synthetic))   

        for i in range(int(math.log2(step_ratio))):
            num_fine = 2 ** (i + 1) * 1024
            grid = gen_grid_up(2 ** (i + 1))
            grid = tf.expand_dims(grid, 0)
            grid_feat = tf.tile(grid, [synthetic.shape[0], 1024, 1])
  
            point_feat = tf.tile(tf.expand_dims(synthetic, 2), [1, 1, 2, 1])
            point_feat = tf.reshape(point_feat, [-1, num_fine, 3])
            global_feat = tf.tile(tf.expand_dims(code, 1), [1, num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)
            feat1=mlp_conv(feat, [128, 64], bn=None, bn_params=None)
            feat1=tf.nn.relu(feat1)
            feat2=contract_expand_operation(feat1, 2)
            feat=feat1+feat2
            with tf.variable_scope('offset', reuse=tf.AUTO_REUSE):
                fine = mlp_conv(feat, [512, 512, 3], bn=None, bn_params=None) + point_feat
            synthetic=fine
            logging.debug('synthetic: {}'.format(synthetic)) 

        return fine

def create_recon_loss(recon, gt):

    loss_ae1 = chamfer(recon, gt)
    add_train_summary('train/ae1_loss', loss_ae1)
    update_loss = add_valid_summary('valid/ae1_loss', loss_ae1)


    visualize_ops = [recon, gt]
    visualize_titles = ['recon output', 'ground truth']
    return loss_ae1, update_loss, visualize_ops, visualize_titles

def create_loss(c1, c2, inputs, coarse, fine, gt, alpha1, alpha2, alpha3):

    model1_l2 = c1
    model2_l2 = c2
    loss_feat = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1_l2, model2_l2), 2), 1, keep_dims=False))
    loss_feat = tf.reduce_mean(loss_feat, keep_dims=False)
    add_train_summary('train/loss_feat', loss_feat)
    update_feat = add_valid_summary('valid/loss_feat', loss_feat)

    loss_coarse = chamfer(coarse, gt)
    add_train_summary('train/coarse_loss', loss_coarse)
    update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)
 

    loss_fine = chamfer(fine, gt)
    add_train_summary('train/fine_loss', loss_fine)
    update_fine = add_valid_summary('valid/fine_loss', loss_fine)

    loss = alpha1 * loss_feat + alpha2 * loss_coarse + alpha3 * loss_fine
    add_train_summary('train/loss', loss)
    update_loss = add_valid_summary('valid/loss', loss)

    visualize_ops = [inputs, coarse, fine, gt]
    visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']
    # return loss_fine, loss, [ update_regularization, update_fine, update_loss], visualize_ops, visualize_titles
    return loss_feat, loss_fine, loss, [update_feat, update_coarse, update_fine, update_loss], visualize_ops, visualize_titles

    
