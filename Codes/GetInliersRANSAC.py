import numpy as np
import cv2
import random
from collections import OrderedDict
from Codes.utils import *
from Codes.EstimateFundamentalMatrix import getFundamentalMatrix

def ransac(pts_src, pts_dst):
    M = 2000
    F_best = -1
    idxs_best = list()
    eps = 1e-4
    for i in range(M):
        idxs = random.sample( list(np.arange(len(pts_src))), 8)
        ps = pts_src[idxs]
        pd = pts_dst[idxs]

        F = getFundamentalMatrix(ps, pd)
        
        src = to_homogeneous(pts_src)
        dst = to_homogeneous(pts_dst)

        dist = dst[:,None,:] @ F[None,:,:] @ src[:,:,None]
        dist = dist.squeeze()
        
        S = np.where(dist <= 1e-4)[0]

        if len(S) > len(idxs_best):
            idxs_best = S
            F_best = F
    
    return F_best, idxs_best




def do_ransac(data_dict):
    new_data_dict = OrderedDict()

    for im_id1 in data_dict.keys():
        new_data_dict[im_id1] = OrderedDict()
        for im_id2 in data_dict[im_id1].keys():
            new_data_dict[im_id1][im_id2] = {"src" : list(), "dst" : list()}
            pts_src = np.array(data_dict[im_id1][im_id2]["src"])
            pts_dst = np.array(data_dict[im_id1][im_id2]["dst"])
            
            F_best, idxs = ransac(pts_src, pts_dst)

            pts_src = pts_src[idxs]
            pts_dst = pts_dst[idxs]

            new_data_dict[im_id1][im_id2]["src"] = pts_src
            new_data_dict[im_id1][im_id2]["dst"] = pts_dst
            new_data_dict[im_id1][im_id2]["F"] = F_best

    return new_data_dict