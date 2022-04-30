import numpy as np
from EstimateFundamentalMatrix import get_F
from collections import OrderedDict
np.random.seed(2)
def get_res(F, p1, p2):
	p1 = np.array([p1[0], p1[1], 1])
	p2 = np.array([p2[0], p2[1], 1])
	return p1@F@p2.T

def ransac_F(kp1, kp2):
	iterations = 4000
	final_in = 0
	for itrs in range(iterations):
		idx = np.random.choice(len(kp1)-1, 8)
		# import pdb; pdb.set_trace()
		pt1 = kp1[idx]
		pt2 = kp2[idx]
		temp_F = get_F(pt1, pt2)
		# temp_F = temp_F/temp_F[2,2]
		inliers = 0
		for i in range(len(kp1)):
			if abs(get_res(temp_F, kp1[i], kp2[i])) < 0.07:
				inliers += 1
		if inliers > final_in:
			F = temp_F
			final_in = inliers
	new_kp1 = []
	new_kp2 = []
	for m in range(len(kp1)):
		if abs(get_res(F, kp1[m], kp2[m])) < 0.07:
			new_kp1.append(kp1[m])
			new_kp2.append(kp2[m])
	# print(len(kp1), len(new_kp1))
	F = get_F(new_kp1, new_kp2)
	# import pdb; pdb.set_trace()
	return F/F[2,2], np.array(new_kp1).astype(int), np.array(new_kp2).astype(int)

def do_ransac(data_dict):
    new_data_dict = OrderedDict()

    for im_id1 in data_dict.keys():
        new_data_dict[im_id1] = OrderedDict()
        for im_id2 in data_dict[im_id1].keys():
            new_data_dict[im_id1][im_id2] = {"src" : list(), "dst" : list()}
            pts_src = np.array(data_dict[im_id1][im_id2]["src"])
            pts_dst = np.array(data_dict[im_id1][im_id2]["dst"])
            
            F_best, pts_src, pts_dst = ransac_F(pts_src, pts_dst)

            # print(f'{im_id1} , {im_id2} --> {len(idxs)} / {len(pts_src)}')

            # pts_src = pts_src[idxs]
            # pts_dst = pts_dst[idxs]

            new_data_dict[im_id1][im_id2]["src"] = pts_src
            new_data_dict[im_id1][im_id2]["dst"] = pts_dst
            new_data_dict[im_id1][im_id2]["F"] = F_best

    return new_data_dict