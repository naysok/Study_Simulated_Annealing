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
    

    def plot_downsampling(self, pts, ds=5):
        
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
        
        pts_ds = rd.plot_downsampling(pts, 1)
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


class CalcVector():

    def calc_distance_2pt(self, pt_0, pt_1):

        ### Calc Distance
        xx = pt_0[0] - pt_1[0]
        yy = pt_0[1] - pt_1[1]
        zz = pt_0[2] - pt_1[2]
        dist = math.sqrt((xx * xx) + (yy * yy) + (zz * zz))

        return dist

cv = CalcVector()


class RayTriangleIntersection():

    ### https://pheema.hatenablog.jp/entry/ray-triangle-intersection

    def __init__(self):
        pass


    def calc_intersection(self, o, d, v0, v1, v2):

        e1 = np.subtract(v1, v0)
        e2 = np.subtract(v2, v0)

        ### https://www.it-swarm.dev/ja/python/python-numpy-machine-epsilon/1041749812/
        kEpsilon = np.finfo(float).eps

        alpha = np.cross(d, e2)
        
        # det = np.dot(e1, alpha)
        det = np.sum(e1 * alpha, axis=1)

        # print("e1.shape : {}".format(e1.shape))
        # print("e2.shape : {}".format(e2.shape))
        # print("alpha.shape : {}".format(alpha.shape))
        # print("det.shape : {}".format(det.shape))

        # intersect_count = np.count_nonzero(det)
    

        ### True = InterSection

        ### (1) Check Parallel
        bool_p = (-kEpsilon > det) | (det > kEpsilon)

        ### Remove (1)
        v0 = v0[bool_p]
        v1 = v1[bool_p]
        v2 = v2[bool_p]
        e1 = e1[bool_p]
        e2 = e2[bool_p]
        alpha = alpha[bool_p]
        det = det[bool_p]
        # print("det.shape (1) : {}".format(det.shape))
        
        
        det_inv = 1.0 / det
        r = np.subtract(o, v0)

        ### (2) Check u-Value in the Domain (0 <= u <= 1)
        # u = np.dot(alpha, r) * det_inv
        u = np.sum(alpha * r, axis=1) * det_inv
        bool_u = (0.0 <= u) & (u <= 1.0)

        ### Remove (2)
        v0 = v0[bool_u]
        v1 = v1[bool_u]
        v2 = v2[bool_u]
        e1 = e1[bool_u]
        e2 = e2[bool_u]
        alpha = alpha[bool_u]
        r = r[bool_u]
        u = u[bool_u]
        det = det[bool_u]
        det_inv = det_inv[bool_u]
        # print("det.shape (2) : {}".format(det.shape))


        beta = np.cross(r, e1)

        ### (3) Check v-Value in the Domain (0 <= v <= 1)
        ### and
        ### Check (u + v = 1)
        # v = np.dot(d, beta) * det_inv
        v = np.sum(d * beta, axis=1) * det_inv
        bool_v = (0.0 <= v) & (u + v <= 1.0)
        
        ### Remove (3)
        v0 = v0[bool_v]
        v1 = v1[bool_v]
        v2 = v2[bool_v]
        e1 = e1[bool_v]
        e2 = e2[bool_v]
        alpha = alpha[bool_v]
        beta = beta[bool_v]
        r = r[bool_v]
        u = u[bool_v]
        v = v[bool_v]
        det = det[bool_v]
        det_inv = det_inv[bool_v]
        # print("det.shape (3) : {}".format(det.shape))

        
        ### (4) Check t_value (t >= 0)
        # t = np.dot(e2, beta) * det_inv
        t = np.sum(e2 * beta, axis=1) * det_inv
        bool_t = 0.0 < t

        ### Remove (4)
        v0 = v0[bool_t]
        v1 = v1[bool_t]
        v2 = v2[bool_t]
        e1 = e1[bool_t]
        e2 = e2[bool_t]
        alpha = alpha[bool_t]
        beta = beta[bool_t]
        r = r[bool_t]
        t = t[bool_t]
        u = u[bool_t]
        v = v[bool_t]
        det = det[bool_t]
        det_inv = det_inv[bool_t]
        # print("det.shape (4) : {}".format(det.shape))

        ### Intersett : True !!
        # intersect_val = [t, u, v]

        ### Barycenrinc_Coordinate >> XYZ
        ### ((1 - u - v) * v0) + (u * v1) + (v * v2)

        new_amp = 1.0 - u - v        
        new_v0 = np.multiply(v0, new_amp[:, np.newaxis])
        new_v1 = np.multiply(v1, u[:, np.newaxis])
        new_v2 = np.multiply(v2, v[:, np.newaxis])
        
        intersect_pos = np.add(np.add(new_v0, new_v1), new_v2)

        if len(intersect_pos) != 0:
            return intersect_pos[0]
        else:
            return np.array([0,0,1000])

rt = RayTriangleIntersection()


class Polygons():


    def project_pts(self, pts):

        ### [ [x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ... [xN yN, zN] ]
        ###
        ### [ [x0, y0,  0], [x1, y1,  0], [x2, y2,  0], ... [xN, yN,  0] ]

        pts_projected = []

        for p in pts:
            pts_projected.append([p[0], p[1], 0])
        
        return pts_projected


    def calc_z(self, pts, x, y):
    
        ### Only Square

        c = int(math.sqrt(len(pts)))

        xx = math.floor(x)
        yy = math.floor(y)

        idx = xx + (yy * c)
        # print("({},{}) : ({},{})".format(x, y, xx, yy))
        # print(idx, idx + 1, idx + c, idx + c + 1)

        base_pt = np.array([x, y, 0.0])
        ray = np.array([0.0, 0.0, 1.0])

        p0 = np.array(pts[idx])
        p1 = np.array(pts[idx + 1])
        p2 = np.array(pts[idx + c])
        p3 = np.array(pts[idx + c + 1])

        """"
        mesh = 
        [
            [v0.X, v0.Y, v0.Z],
            [v1.X, v1.Y, v1.Z], 
            [v2.X, v2.Y, v2.Z]
        ]
        """

        ### meshes
        meshes = [[p0, p1, p3], [p0, p3, p2]]
        meshes = ut.transpose_nested_list(meshes)

        ### Ray-Triangle
        o = np.array(base_pt).reshape(1, 3)
        d = np.array([0.0, 0.0, 1.0]).reshape(1, 3)
        v0 = np.array(meshes[0])
        v1 = np.array(meshes[1])
        v2 = np.array(meshes[2])

        inter_pt = rt.calc_intersection(o, d, v0, v1, v2)
        # print(inter_pt)

        dist = cv.calc_distance_2pt(inter_pt, base_pt)

        
        PLOT = False

        if PLOT:

            ### Plot
            tmp = np.stack([p0, p1, p2, p3, base_pt])
            tmp_t = tmp.T 

            rd.plot_scaling(tmp_t[0], tmp_t[1], tmp_t[2])

            poly_0 = np.array([p0, p1, p3, p0])
            poly_1 = np.array([p0, p3, p2, p0])

            poly_0 = poly_0.T
            poly_1 = poly_1.T

            plt.plot(poly_0[0], poly_0[1], poly_0[2], color="red")
            plt.plot(poly_1[0], poly_1[1], poly_1[2], color="red")

            plt.plot(base_pt[0], base_pt[1], base_pt[2], color="blue", marker='x', markersize=2)

            plt.plot(inter_pt[0], inter_pt[1], inter_pt[2], color="blue", marker='x', markersize=2)


            plt.show()
        

        return inter_pt, dist

pg = Polygons()


class SASolver():

    def cost_f(self, pts, x, y):
        
        ### CalcPolygon
        inter_pt, dist = pg.calc_z(pts, x, y)
        # print(dist)

        return inter_pt, dist
    

    def annealing_optimize(self, pts, T=10000, cool=0.99, step=1.0):

        log_ = []

        # vec = random.randint(-1, 2)
        x = random.randint(5, 35)
        y = random.randint(5, 35)
        
        count = 0

        while T > 0.0001:

            dir_ = random.random()

            if dir_ < 0.5:
                dir_ = ((dir_ * 2) - 0.5) * step
                
                # if (x + dir_ > 0) and (x + dir_) < 40:
                #     new_x = x + dir_
                #     new_y = y
                # else:
                #     new_x = x
                #     new_y = y
                new_x = x + dir_
                new_y = y

            else:
                dir_ = (((dir_ - 0.5) * 2) - 0.5) * step

                # if (y + dir_ > 0) and (y + dir_) < 40:
                    # new_x = x
                    # new_y = y + dir_
                # else:
                    # new_x = x
                    # new_y = y
                
                new_x = x
                new_y = y + dir_

            # dir_ = (dir_ - 0.5) * step
            # new_vec = vec + dir_
 

            ### Clac Value
            # new_cost = self.cost_f(new_vec)
            # old_cost = self.cost_f(vec)

            new_inter_pt, new_cost = self.cost_f(pts, new_x, new_y)
            new_inter_pt, old_cost = self.cost_f(pts, x, y)

            ### Probability
            p = pow(math.e, (-1) * abs(new_cost - old_cost) / T)
            # print("{} : {}".format(T, new_vec))
            print("{} : {}/{}".format(T, new_x, new_y))
            # print(new_cost)


            ### Update
            if (new_cost < old_cost) or (random.random() < p):
                # vec = new_vec
                x = new_x
                y = new_y

                # log_.append(new_inter_pt)

            log_.append(new_inter_pt)


            T = T * cool

            # print(count)
            count = count + 1
        
        # print("Count : {}".format(count))

        return x, y, log_


sa = SASolver()




prj_path = "C:\\Users\\SJ005\\Documents\\_private_dev\\Study_Simulated_Annealing\\"
file_path = prj_path + "_data\\" + "surface_2.txt"






pts = ut.read_data(file_path)

x, y, log_ = sa.annealing_optimize(pts)

print("RESULT : {},{}".format(x, y))


# print(log_)
log_ = np.array(log_)
log_ = log_.tolist()

# print(log_)
# print(type(log_))

export_log = []

for e in log_:
    export_log.append("{},{},{}".format(e[0], e[1], e[2]))



prj_path = "C:\\Users\\SJ005\\Documents\\_private_dev\\Study_Simulated_Annealing\\"
result_path = prj_path + "_result\\" + "surface_2.txt"

with open(result_path, mode='w') as f:
    f.write('\n'.join(export_log))



"""


def cost_f(x):
    return (3*(x**4)) - (5*(x**3)) + (2*(x**2))



def annealing_optimize(T=10000, cool=0.99, step=1):

    vec = random.randint(-1, 2)
    count = 0

    while T > 0.0001:

        dir_ = random.random()
        dir_ = (dir_ - 0.5) * step

        new_vec = vec + dir_

        ### Clac Value
        new_cost = cost_f(new_vec)
        old_cost = cost_f(vec)

        ### Probability
        p = pow(math.e, (-1) * abs(new_cost - old_cost) / T)
        print("{} : {}".format(T, new_vec))

        if (new_cost < old_cost) or (random.random() < p):
            vec = new_vec
        
        T = T * cool

        # print(count)
        count = count + 1
    
    print("Count : {}".format(count))

    return vec





print('Result : {:f}'.format(annealing_optimize()))

"""
