import numpy as np
from torch.utils.data import Dataset

eps = 0.001

class test_data_plot(Dataset):
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

