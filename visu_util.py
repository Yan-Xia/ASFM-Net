# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
import matplotlib
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='jet', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_feature_pca(filename, feature1, feature2):
    pca = PCA(n_components=2)
    feature1_pca = pca.fit_transform(feature1)
    feature2_pca = pca.fit_transform(feature2)
    plt.scatter(feature1[:, 0], feature1[:, 1], cmap='red',
                label='feature_by_auto_encoder_1')
    plt.scatter(feature2[:, 0], feature2[:, 1],
                cmap='blue', label='feature_by_auto_encoder_2')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()


def plot_cd_loss(filename, x, y1):
    
    figure1 = plt.figure(figsize=(9,9))
    ax1 = figure1.add_subplot(1,1,1)
    ax1.scatter(x,y1,marker = 'x',color='red',s=40, label='Chamfer')
    ax1.set_ylim((0.010,0.018))
    ax1.set_xlabel("Train Step")
    ax1.set_ylabel("Charmfer Distance")
    # plt.scatter(x,y2,marker = 'o',color='green',s=40, label='Earth Mover')
    figure1.savefig(filename)
    plt.close(figure1)

def plot_emd_loss(filename, x, y2):
    figure1 = plt.figure(figsize=(9,9))
    ax1 = figure1.add_subplot(1,1,1)
    ax1.scatter(x,y2,marker = 'x',color='red',s=40, label='Earth Mover')
    ax1.set_xlabel("Train Step")
    ax1.set_ylabel("Earth Mover Distance")
    # plt.scatter(x,y2,marker = 'o',color='green',s=40, label='Earth Mover')
    figure1.savefig(filename)
    plt.close(figure1)


