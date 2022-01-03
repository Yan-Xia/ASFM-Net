#2020/5/14

import argparse
import numpy as np
import os
from io_util import read_pcd, save_pcd
from data_util import resample_pcd


def resample_testdata(args):
    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    for i, model_id in enumerate(model_list):
        partial = read_pcd(os.path.join(args.data_dir, 'partial', '%s.pcd' % model_id))   
        complete = read_pcd(os.path.join(args.data_dir, 'complete', '%s.pcd' % model_id))
        complete = resample_pcd(complete,args.num_gt_points)
        synset_id, model_id = model_id.split('/')
        os.makedirs(os.path.join(args.results_dir, 'partial', synset_id), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, 'complete', synset_id), exist_ok=True)
        save_pcd(os.path.join(args.results_dir, 'partial', '%s' %synset_id, '%s.pcd' % model_id), complete)
        save_pcd(os.path.join(args.results_dir, 'complete', '%s' %synset_id, '%s.pcd' % model_id), complete)
    print('testdata saved')
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path', default='data/shapenet/test.list')
    parser.add_argument('--data_dir', default='data/shapenet/test')
    parser.add_argument('--results_dir', default='data/shapenet/test4096')
    parser.add_argument('--num_gt_points', type=int, default=4096)
    args = parser.parse_args()

    resample_testdata(args)
