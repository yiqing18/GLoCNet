import cv2
import numpy as np
import os
import scipy.io as scio

def choice(x):
    smallscalar = [2,5,10,15,-2,-5,-10,-15]
    bigscalar = [10, 20, 30, 40, 50,-10,-20,-30,-40,-50]
    x_min, x_max = x[:,0].min(), x[:,0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()
    source = np.stack(([x_min,y_min],[x_min,y_max],[x_max,y_min],[x_max,y_max]))
    new_points = []
    for i, j in enumerate(source):
        nx = j[0] + np.random.choice(smallscalar,1)
        ny = j[1] + np.random.choice(bigscalar,1)

        new_points.append((nx,ny))
    target = np.array(new_points,np.float32)
    source = np.array(source,np.float32)
    return source,target

def perspective_rotate(ps, r):
    pts = np.float32(ps).reshape([-1, 2])  
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.matmul(r,pts).T
    target_point[:,0] = target_point[:,0]/target_point[:,2]
    target_point[:, 1] = target_point[:, 1] / target_point[:, 2]
    return target_point[:,:2]

# def choice4(data_path):
#     img_pair = np.random.choice(os.listdir(data_path))
#     data = scio.loadmat(os.path.join(data_path,img_pair))
#     gt = data['GroundTruth']
#     point1 = data['point1']
#     point2 = data['point2']
#     source, target = choice(point2)
#     matrix = cv2.getPerspectiveTransform(source, target)
#     kp2_rotated = perspective_rotate(point2, matrix)

#     return point1, kp2_rotated, gt

def choice3(img):
    h, w = img.shape[0:2]
    N = 5
    pad_pix = 50
    points = []
    dx = int(w/ (N - 1))
    for i in range( N):
        points.append((dx * i,  pad_pix))
        points.append((dx * i, pad_pix + h))

    source = np.array(points, np.int32)
    source = source.reshape(1, -1, 2)

    bigscalar = [10, 20, 30, 40, 50,-10,-20,-30,-40,-50]

    newpoints = []
    for i in range(N):

        nx = points[i][0] + np.random.choice(bigscalar, 1)
        ny = points[i][1] + np.random.choice(bigscalar, 1)

        newpoints.append((nx, ny))

    target = np.array(newpoints, np.int32)
    target = target.reshape(1, -1, 2)
    matches = []
    for i in range(1, 2*N + 1):
        matches.append(cv2.DMatch(i, i, 0))

    return source, target, matches

