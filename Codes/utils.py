import numpy as np
import cv2
from scipy.spatial.distance import cdist

def to_homogeneous(points):
    ones = np.ones((points.shape[:-1]))
    ones = np.expand_dims(ones, -1)
    points = np.concatenate([points, ones], axis=-1)
    return points

def to_inhomogeneous(points):
    points[:,0] = points[:,0] / points[:,2]
    points[:,1] = points[:,1] / points[:,2]
    return points[:,:2]


def get_norm_matrix(data):

    assert len(data.shape) == 2, "shape mismatch"

    x, y = data[:, 0], data[:, 1]

    N = data.shape[0]

    x_mean, y_mean = x.mean(), y.mean()
    x_var, y_var = x.var(), y.var()
    
    # Form rescaling matrix so that data points will lie
    # sqrt(2) from the origin on average.
    s_x, s_y = np.sqrt(2. / x_var), np.sqrt(2. / y_var)
    
    norm_matrix = np.array([[s_x,  0., -s_x * x_mean],
                            [ 0., s_y, -s_y * y_mean],
                            [ 0.,  0.,            1.]])

    return norm_matrix

def draw_points(img, points, color):
    for p in points:
        img = cv2.circle(img, (int(p[0]), int(p[1])), 5, color, -1)
    return img

def solve_svd(A):
    U,S,V = np.linalg.svd(A)
    idx = np.argmin(S)
    return V[idx]

def get_common_world_points(X, data_dict, registered_images, registered_image_points, im_id2):

    best_im_id = None
    best_x_pts = list()
    best_X_pts = list()

    for im_id, pts in zip(registered_images, registered_image_points):
        
        src = data_dict[im_id][im_id2]["src"]
        dst = data_dict[im_id][im_id2]["dst"]
        
        dist = cdist(src, pts)
        id1, id2 = np.where(dist == 0)
        
        if len(id1) > 0:
            best_x_pts += dst[id1].tolist()
            best_X_pts += X[id2].tolist()
    
    
    best_X_pts, idxs = np.unique(best_X_pts, axis=0, return_index=True)
    best_x_pts = np.array(best_x_pts)[idxs]

    assert len(best_X_pts) == len(best_x_pts)

    print(f"Found --> {len(best_x_pts)}")

    return best_X_pts, best_x_pts