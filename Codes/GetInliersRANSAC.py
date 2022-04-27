import numpy as np
import cv2

def do_ransac(data_dict):
    new_data_dict = dict()
    for im_id1 in data_dict.keys():
        new_data_dict[im_id1] = dict()
        for im_id2 in data_dict[im_id1].keys():
            new_data_dict[im_id1][im_id2] = {"src" : list(), "dst" : list()}
            pts_src = np.array(data_dict[im_id1][im_id2]["src"])
            pts_dst = np.array(data_dict[im_id1][im_id2]["dst"])
            
            h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,5.0)

            idxs = np.where(status==1)[0]

            pts_src = pts_src[idxs]
            pts_dst = pts_dst[idxs]

            new_data_dict[im_id1][im_id2]["src"] = pts_src
            new_data_dict[im_id1][im_id2]["dst"] = pts_dst

    return new_data_dict