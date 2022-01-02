#Train the auto-encoder for complete point clouds

import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
import sys
from tqdm import tqdm
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))

def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(
        tf.float32, (args.batch_size, args.num_gt_points, 3), 'inputs')
    npts = tf.constant(args.num_gt_points, shape=[32,], name='num_points')


    model = importlib.import_module('.%s' % args.model, 'models')
    features = model.create_ae1_encoder(inputs_pl, npts)
    recon = model.create_ae1_decoder(features)
    model_loss, update, visualize_ops, visualize_titles = model.create_recon_loss(recon, inputs_pl)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')
    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model_loss, global_step)

    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_gt_points, args.num_gt_points, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_gt_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()
    

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=1)

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
    else:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % args.log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')      # log of arguments
        os.system('cp models/%s.py %s' % (args.model, args.log_dir))        # bkp of model def
        os.system('cp %s.py %s' % (args.train_file, args.log_dir))          # bkp of train file
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)
    min_loss = 1.0
    for step in tqdm(range(init_step+1, args.max_step+1), leave=False):
        epoch = step * args.batch_size // num_train + 1       
        ids, _, _, gt = next(train_gen)
        start = time.time()
        feed_dict = {inputs_pl: gt, is_training_pl: True}
        _, loss, summary = sess.run([train_op, model_loss, train_summary], feed_dict=feed_dict)
        total_time += time.time() - start
        writer.add_summary(summary, step)
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0
        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            total_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in tqdm(range(num_eval_steps), leave=True):
                start = time.time()
                ids, _, _, gt = next(valid_gen)
                feed_dict = {inputs_pl: gt, is_training_pl: False}
                loss, _ = sess.run([model_loss, update], feed_dict=feed_dict)
                total_loss += loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0
            if  total_loss / num_eval_steps <min_loss :
                min_loss = total_loss/num_eval_steps
                saver.save(sess, os.path.join(args.log_dir, 'model'))
                print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))
                # all_pcds = sess.run(visualize_ops, feed_dict=feed_dict)
                # for i in range(0, args.batch_size, args.visu_freq):
                #     plot_path = os.path.join(args.log_dir, 'plots',
                #                             'epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
                #     pcds = [x[i] for x in all_pcds]
                #     plot_pcd_three_views(plot_path, pcds, visualize_titles)

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet/valid.lmdb')
    parser.add_argument('--log_dir', default='log/ae1')
    parser.add_argument('--model', default='asfm_net')
    parser.add_argument('--train_file', default='pretrain')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_gt_points', type=int, default=4096)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=3000)
    parser.add_argument('--visu_freq', type=int, default=5)
    args = parser.parse_args()

    train(args)
