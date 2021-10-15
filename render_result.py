import random

import math
from matplotlib import colors

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy.linalg as linalg
import time


class Util():


    def file_open(self, file_path):
        with open(file_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
            # print(l_strip)

        pts = []
        for e in l_strip:
            x, y, z = e.split(",")
            pts.append([float(x), float(y), float(z)])
        
        pts = np.array(pts)
        return pts


    def partition_list(self, l):

        ### Only Square

        c = int(math.sqrt(len(l)))
        # print(c)

        result = []
        for i in range(c):
            sub = []
            for j in range(c):
                sub.append(l[c*i + j])
            result.append(sub)
        return result
    

    def transpose_nested_list(self, l_2d):
        return [list(x) for x in zip(*l_2d)]


    def read_data(self, file_path):

        ### pts = 1600
        ### pts_grid = 40x40

        pts = self.file_open(file_path)
        # pts_grid = self.partition_list(pts)

        return pts

ut = Util()


class Render():

    def plot_scaling(self, X, Y, Z):

        ### https://qiita.com/ae14watanabe/items/71f678755525d8088849

        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() * 0.5

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    

    def plot_downsampling(self, pts, ds=2):
        
        ### Only Square

        c = int(math.sqrt(len(pts)))
        # print(c)

        plot_pts = []
        for i in range(c):
            if i%ds == 0:
                for j in range(c):
                    if j%ds == 0:
                        plot_pts.append(pts[c*i + j])
        plot_pts = np.array(plot_pts)
        return plot_pts


    def plot_surface(self, pts):
        
        pts_ds = self.plot_downsampling(pts, 1)
        pts_ds = pts_ds.T

        rd.plot_scaling(pts_ds[0], pts_ds[1], pts_ds[2])
        
        ### Dot
        ax.scatter(pts_ds[0], pts_ds[1], pts_ds[2], s = 2)

        ### Surface
        # XX = np.array(ut.partition_list(pts_ds[0].tolist()))
        # YY = np.array(ut.partition_list(pts_ds[1].tolist()))
        # ZZ = np.array(ut.partition_list(pts_ds[2].tolist()))
        # ax.plot_surface(XX, YY, ZZ, cmap="coolwarm")

rd = Render()



### Plot


prj_path = "C:\\Users\\SJ005\\Documents\\_private_dev\\Study_Simulated_Annealing\\"
file_path = prj_path + "_data\\" + "surface_2.txt"
result_path = prj_path + "_result\\" + "surface_2.txt"


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


pts = ut.read_data(file_path)
log_ = ut.read_data(result_path)
log_ = log_.T


rd.plot_surface(pts)

plt.plot(log_[0], log_[1], log_[2], color="blue")







plt.show()