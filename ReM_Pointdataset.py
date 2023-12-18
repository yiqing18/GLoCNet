import numpy as np
from torch.utils.data import Dataset
import cv2
import random
import os
import datetime
import TPS as tps

eps = 0.001

nfeatures = 1000
cluster_path = 'cluster_result/'
eps = 0.001


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    # M[0, 2] += (w / 2) - cX
    # M[1, 2] += (h / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w,h)), M

def affine_rotate(ps, r,t):
    pts = np.float32(ps).reshape([-1, 2])  
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    # target_point = np.dot(m, pts)
    target_point = np.matmul(r,pts)+t
    # target_point = [[int(target_point[0][x]),int(target_point[1][x])] for x in range(len(target_point[0]))]
    return np.array(target_point[:2]).T

def perspective_rotate(ps, r):
    pts = np.float32(ps).reshape([-1, 2])  
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    # target_point = np.dot(m, pts)
    target_point = np.matmul(r,pts).T
    target_point[:,0] = target_point[:,0]/target_point[:,2]
    target_point[:, 1] = target_point[:, 1] / target_point[:, 2]
    return target_point[:,:2]

class train_data_preprocess(Dataset):

    def __init__(self, img_path, nfeatures):
        self.img_path = img_path
        self.nfeatures = nfeatures
        self.files = []
        self.files += [os.path.join(self.img_path,i) for i in os.listdir(self.img_path)]
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        self.ratio = [0.3,0.5,0.7]
        seed = datetime.datetime.now().second + datetime.datetime.now().microsecond
        self.thre= 1
        self.sift_threshold = 1

    def __len__(self):
        return len(self.files)

    def repeat_removal(self, kp):
        temp = np.zeros([len(kp), 2])
        for i in range(len(kp)):
            temp[i] = kp[i].pt
        _, index = np.unique(temp, return_index=True, axis=0)
        return index

    def point_normalize(self,point):
        min_x = np.min(point[:,0])
        min_y = np.min(point[:,1])
        max_x = np.max(point[:, 0])
        max_y = np.max(point[:, 1])
        # Z
        if point.shape[1] > 2:
            min_z = np.min(point[:, 2])
            max_z = np.max(point[:, 2])

            min = np.stack((min_x, min_y, min_z))
            max = np.stack((max_x, max_y, max_z))
        else:
            min = np.stack((min_x, min_y))
            max = np.stack((max_x, max_y))
        new_point = (point-min)/(max-min)
        return new_point


    def theta_cluster(self, point1, point2):
        w_max = np.max(point1)
        w_min = np.min(point1)
        point2v = point2.copy()
        point2v[:, 0] = point2[:, 0] + (w_max - w_min)
        vec_direc = point2v - point1
        vec_theta = np.arctan((vec_direc[:, 1] / (vec_direc[:, 0]+eps))) * 180 / np.pi

        theta_y_conti = np.linspace(0, vec_theta.shape[0], vec_theta.shape[0], dtype=int)


        vec_len = np.sqrt(np.sum(vec_direc ** 2, axis=1))

        X = np.stack((theta_y_conti, vec_len, vec_theta)).T
        XN = self.point_normalize(X)

        return XN

    def __getitem__(self, item):
        src = self.files[item]
        img = cv2.imread(src)[:, :, [2, 1, 0]]
        kp1, desc1 = self.sift.detectAndCompute(img, None)
        index1 = self.repeat_removal(kp1)
        kp1 = np.array(kp1)[index1]
        desc1 = np.array(desc1)[index1]
        kp1_np = np.array([k.pt for k in kp1])

        sopeator = [0,1] # 0:perspective,1:affine
        sop = np.random.choice(sopeator)

        if sop ==0:
            source, target = tps.choice(kp1_np)
            matrix = cv2.getPerspectiveTransform(source, target)
            kp1_rotated = perspective_rotate(kp1_np, matrix)
        if sop==1:

            R = np.mat(np.random.rand(2, 2))
            T = np.mat(np.random.rand(2, 1))

            R = np.column_stack((R, [0, 0]))
            R = np.row_stack((R, [0, 0, 1]))
            T = np.row_stack((T, 0))

            u, s, v = np.linalg.svd(R)
            R = u * v
            if np.linalg.det(R) < 0:
                v[2, :] *= -1
                R = u * v
            kp1_rotated = affine_rotate(kp1_np, R,T)
        kp2_np = np.zeros_like(kp1_np)
        ratio = np.random.choice(self.ratio,1)
        rotate_part_index = np.array(
            random.sample(list(np.arange(0, len(kp1_np), dtype=int)), int(ratio * len(kp1_np))))
        rotate_rest_index = np.setdiff1d(np.array(np.arange(0,len(kp1_np),dtype=int)), rotate_part_index)

        kp2_np[rotate_part_index] = np.array(kp1_rotated[rotate_part_index,:2])
        choice_array = np.hstack(np.array(kp1_rotated[rotate_rest_index,:2]))
        kp2_rest_x = np.random.choice(choice_array, len(rotate_rest_index))
        kp2_rest_y = np.random.choice(choice_array, len(rotate_rest_index))
        kp2_np[rotate_rest_index] = np.vstack((kp2_rest_x, kp2_rest_y)).T
        matches = np.zeros((len(kp1_np), 1))
        matches[rotate_part_index, :] = 1  # inlier = 1, outlier = 0

        kp1_npn = self.point_normalize(kp1_np)
        kp2_npn = self.point_normalize(kp2_np)

        theta_distri = self.theta_cluster(kp1_npn, kp2_npn)


        return {'kp1':kp1_npn, 'kp2':kp2_npn, 'matches': matches, 'M':np.array([]),'theta_dis':theta_distri}


class test_data_preprocess(Dataset):
    def __init__(self, data_config, gt):
        self.point1 = data_config["point1"]
        self.point2 = data_config["point2"]
        self.gt = gt

    def point_normalize(self,point):
        min_x = np.min(point[:, 0])
        min_y = np.min(point[:, 1])
        max_x = np.max(point[:, 0])
        max_y = np.max(point[:, 1])
        if point.shape[1]>2:
            min_z = np.min(point[:, 2])
            max_z = np.max(point[:, 2])

            min = np.stack((min_x, min_y,min_z))
            max = np.stack((max_x, max_y,max_z))
        else:
            min = np.stack((min_x, min_y))
            max = np.stack((max_x, max_y))

        new_point = (point - min) / (max - min)

        return new_point

    def theta_cluster(self, point1, point2):
        w_max = np.max(point1)
        w_min = np.min(point1)
        h_max = np.max(point1[:, 1])
        h_min = np.min(point1[:, 1])
        point2v = point2.copy()
        point2vv = point2.copy()
        point2v[:, 0] = point2[:, 0] + (w_max - w_min)
        point2vv[:, 1] = point2[:, 1] + (h_max - h_min)
        vec_direc_y = point2v - point1
        vec_theta_y = np.arctan((vec_direc_y[:, 1] / vec_direc_y[:, 0]+eps)) * 180 / np.pi
        theta_y_conti = np.linspace(0, vec_theta_y.shape[0], vec_theta_y.shape[0], dtype=int)

        ## vec_length
        vec_len = np.sqrt(np.sum(vec_direc_y**2,axis=1))

        X_y = np.stack((theta_y_conti, vec_len, vec_theta_y)).T
        XN_y = self.point_normalize(X_y)

        return XN_y

    def __getitem__(self, index):

        kpts1 = self.point_normalize(self.point1)
        kpts2 = self.point_normalize(self.point2)
        vec_the_y= self.theta_cluster(kpts1, kpts2)

        
        return {'kp1': kpts1, 'kp2': kpts2, 'theta_dis': vec_the_y,
                'M': float(1), 'matches': (self.gt / 1.)}


    def __len__(self):
        return 1

