
import argparse
import csv
import importlib
import models
import numpy as np
import os
import tensorflow as tf
import time
import sys
from io_util import read_pcd, save_pcd
from tf_util import chamfer, earth_mover
from visu_util import plot_pcd_three_views, plot_cd_loss
from data_util import pad_cloudN
from tqdm import tqdm
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))


def test(args):
    file_name = os.listdir(args.log_dir)
    checkpoint_step_list=[os.path.splitext(checkpoint_name)[0] for checkpoint_name in file_name if os.path.splitext(checkpoint_name)[1]=='.index'] 

    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    inputs_pad = tf.placeholder(tf.float32, (1, args.num_input_points, 3))
    model = importlib.import_module('.%s' % args.model, 'models')
    features = model.create_ae2_encoder(inputs, npts)
    coarse = model.create_ae2_decoder(features)
    fine=model.create_refiner(features, inputs_pad, coarse, args.step_ratio, num_extract=512)

    cd_op = chamfer(fine, gt)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    plot_step = []
    cd_loss_avg = []

    for j,checkpoint_step in enumerate(checkpoint_step_list):
        plot_step.append(int(checkpoint_step[6:]))
        saver.restore(sess, os.path.join(args.log_dir, checkpoint_step))
        results_dir = os.path.join(args.results_dir,checkpoint_step)
        os.makedirs(results_dir, exist_ok=True)
        csv_file = open(os.path.join(results_dir, 'results.csv'), 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'chamfer loss'])

        with open(args.list_path) as file:
            model_list = file.read().splitlines()
        total_time = 0
        total_cd = 0
        cd_per_cat = {}
        for i, model_id in enumerate(tqdm(model_list)):
            partial = read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))
            complete = read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id))
            partial_pad = pad_cloudN(partial, args.num_input_points)
            start = time.time()
            coarse_out, fine_out = sess.run([coarse, fine], feed_dict={inputs: [partial], inputs_pad:[partial_pad], npts: [partial.shape[0]]})
            total_time += time.time() - start
            cd = sess.run(cd_op, feed_dict={fine: fine_out, gt: [complete]})
            total_cd += cd
            writer.writerow([model_id, cd])

            synset_id, model_id = model_id.split('/')
            if not cd_per_cat.get(synset_id):
                cd_per_cat[synset_id] = []
            cd_per_cat[synset_id].append(cd)

            if i % args.plot_freq == 0:
                os.makedirs(os.path.join(results_dir, 'plots', synset_id), exist_ok=True)
                plot_path = os.path.join(results_dir, 'plots', synset_id, '%s.png' % model_id)
                plot_pcd_three_views(plot_path, [partial, coarse_out[0], fine_out[0], complete],
                                 ['input', 'coarse output', 'fine output', 'ground truth'],
                                 'CD %.4f ' % (cd),
                                 [5, 5, 0.5, 0.5])
            if args.save_pcd:
                os.makedirs(os.path.join(results_dir, 'pcds', synset_id), exist_ok=True)
                save_pcd(os.path.join(results_dir, 'pcds', '%s.pcd' % model_id), completion[0])
        csv_file.close()
        print('Result for :%s'% checkpoint_step)
        print('Average time: %f' % (total_time / len(model_list)))
        cd_loss_avg.append(total_cd / len(model_list))
        print('Average Chamfer distance: %f' % (total_cd / len(model_list)))
        print('Chamfer distance per category')
        for synset_id in cd_per_cat.keys():
            print(synset_id, '%f' % np.mean(cd_per_cat[synset_id]))


    plot_path_cd = os.path.join(args.results_dir, 'CD_Loss_plot.png')
    plot_cd_loss(plot_path_cd, plot_step, cd_loss_avg)
    sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='data/shapenet/test.list')
    parser.add_argument('--data_dir', default='data/shapenet/test4096')
    parser.add_argument('--model', default='asfm_net')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--log_dir', default='log/asfm')
    parser.add_argument('--results_dir', default='results/asfm/shapenet')
    parser.add_argument('--num_gt_points', type=int, default=4096)
    parser.add_argument('--num_input_points', type=int, default=4096)
    parser.add_argument('--step_ratio', type=int, default=4)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    args = parser.parse_args()

    test(args)
