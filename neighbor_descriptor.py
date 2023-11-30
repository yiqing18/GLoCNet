import numpy as np
from sklearn.neighbors import KDTree


def gauss(x, sigma, c):
    return np.exp(-np.square(x-c)/(2*sigma**2))

def edist(x,y):
    return np.sqrt(np.power((x-y),2))

def cw_rotate(x, y, rx, ry, ang):
    new_x = (x-rx) * np.cos(ang) + (y-ry) * np.sin(ang) + rx
    new_y = -(x-rx) * np.sin(ang) + (y-ry) * np.cos(ang) + ry
    return new_x, new_y

def axis_transform(x):
    ## x:[1,N,7,2]
    eps = 0.000001
    xc = x
    new_xc = np.zeros_like(xc)
    delta = xc[:, 1] - xc[:, 0]
    # theta = np.arctan(delta[:,1]/(delta[:,0] + eps)) * 180 / np.pi
    theta = np.arctan(delta[:, 1] / (delta[:, 0] + eps))
    theta[list(set(np.where(theta >= 0)[0]).intersection(np.where(delta[:, 1] <= 0)[0]))] = theta[list(
        set(np.where(theta >= 0)[0]).intersection(np.where(delta[:, 1] <= 0)[0]))] + np.pi

    theta[list(set(np.where(theta <= 0)[0]).intersection(np.where(delta[:, 1] <= 0)[0]))] = theta[list(
        set(np.where(theta <= 0)[0]).intersection(np.where(delta[:, 1] <= 0)[0]))] + 2 * np.pi

    theta[list(set(np.where(theta <= 0)[0]).intersection(np.where(delta[:, 1] > 0)[0]))] = theta[list(
        set(np.where(theta <= 0)[0]).intersection(np.where(delta[:, 1] > 0)[0]))] + np.pi
    for i, j in enumerate(delta):
        new_xc[i, :, 0], new_xc[i, :, 1] = cw_rotate(xc[i, :, 0], xc[i, :, 1], xc[i, 0, 0], xc[i, 0, 1], theta[i])

    return new_xc

def select_neighbor(p1_raw, p2_raw,selec_index,k):
    K0 = k
    K1 = k
    K2 = k
    point1 = p1_raw[selec_index]
    point2 = p2_raw[selec_index]
    #round1
    tree1 = KDTree(point1)
    tree2 = KDTree(point2)
    Neighbor_X = tree1.query(point1, k = K0)
    Neighbor_Y = tree2.query(point2, k = K0)
    bb = np.append(Neighbor_X[1], Neighbor_Y[1], axis = 1)
    sort_bb = np.diff(np.sort(bb),axis=1)
    dd = (sort_bb == (np.zeros(sort_bb.shape))).astype(float)
    count_both_point = np.sum(dd, axis=1,keepdims=True) - 1
    pp_inlier = count_both_point / K0
    idx = np.where(pp_inlier>0.1)[0]

    #round2
    tree1 = KDTree(point1[idx])
    tree2 = KDTree(point2[idx])
    Neighbor_X = tree1.query(point1, k=K1)
    Neighbor_Y = tree2.query(point2, k=K1)
    Neighbor_X = idx[Neighbor_X[1]]
    Neighbor_Y = idx[Neighbor_Y[1]]
    bb = np.append(Neighbor_X, Neighbor_Y, axis=1)
    sort_bb = np.diff(np.sort(bb), axis=1)
    dd = (sort_bb == (np.zeros(sort_bb.shape))).astype(float)
    count_both_point = np.sum(dd, axis=1, keepdims=True) - 1
    pp_inlier = count_both_point / K0
    idx = np.where(pp_inlier > 0.3)[0]

    #round3
    tree1 = KDTree(point1[idx])
    tree2 = KDTree(point2[idx])
    Neighbor_X0 = tree1.query(point1, k=K2)
    Neighbor_Y0 = tree2.query(point2, k=K2)
    Neighbor_X = idx[Neighbor_X0[1]]
    Neighbor_Y = idx[Neighbor_Y0[1]]
    bb = np.append(Neighbor_X, Neighbor_Y, axis=1)
    sort_bb = np.diff(np.sort(bb), axis=1)
    dd = (sort_bb == (np.zeros(sort_bb.shape))).astype(float)
    count_both_point = np.sum(dd, axis=1, keepdims=True) - 1
    pp_inlier = count_both_point / K0
    idx = np.where(pp_inlier > 0.5)[0]
    raw_idx = selec_index[idx]

    tree1 = KDTree(point1)  
    tree2 = KDTree(point2)
    Neighbor_X0 = tree1.query(p1_raw, k=K2)  
    Neighbor_Y0 = tree2.query(p2_raw, k=K2)
    Neighbor_X = selec_index[Neighbor_X0[1]]  
    Neighbor_Y = selec_index[Neighbor_Y0[1]]
    bb = np.append(Neighbor_X, Neighbor_Y, axis=1)
    sort_bb = np.diff(np.sort(bb), axis=1)
    dd = (sort_bb == (np.zeros(sort_bb.shape))).astype(float)
    count_both_point = np.sum(dd, axis=1, keepdims=True) - 1
    pp_inlier = count_both_point / K0
    iii = np.where(pp_inlier > 0)[0]

    Neighbor_X0 = tree1.query(p1_raw[iii], k=K2)
    Neighbor_Y0 = tree2.query(p2_raw[iii], k=K2)
    Neighbor_X = selec_index[Neighbor_X0[1]]
    Neighbor_Y = selec_index[Neighbor_Y0[1]]
    NeiX = axis_transform(p1_raw[Neighbor_X])
    NeiY = axis_transform(p2_raw[Neighbor_X])

    return iii, NeiX, NeiY
