#!/bin/bash
source activate asfm-net
cd pc_distance
# make clean
make
cd ..
cd tf_ops
cd grouping
# rm *.o
# rm *.so
./tf_grouping_compile.sh
cd ..
cd sampling
# rm *.o
# rm *.so
./tf_sampling_compile.sh

cd ..
cd ..
python pretrain.py
python train.py
python test.py