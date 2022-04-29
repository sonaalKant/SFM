import numpy as np
from scipy.optimize import least_squares, curve_fit
from Codes.utils import *

def ReprojectionError(X, pts_src, pts_dst, P0, P1):

    X = X.reshape(len(X)//3, 3)
    X = to_homogeneous(X)
    
    # import pdb;pdb.set_trace()
    camera1 = (pts_src[:,0] - P0[0,:]@X.T / (P0[2,:]@X.T))**2 + (pts_src[:,1] - P0[1,:]@X.T / (P0[2,:]@X.T))**2
    camera2 = (pts_dst[:,0] - P1[0,:]@X.T / (P1[2,:]@X.T))**2 + (pts_dst[:,1] - P1[1,:]@X.T / (P1[2,:]@X.T))**2
    
    # import pdb;pdb.set_trace()
    return camera1 + camera2


def nonlinear_triangulation(K, C0, R0, C1, R1, pts_src, pts_dst, X0):
    I = np.eye(3)
    P0 = np.dot(K, np.dot(R0, np.hstack((I, -C0))))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))

    X_final = list()
    # for X, src, dst in zip(X0, pts_src, pts_dst):
    # X_new = least_squares(ReprojectionError, X0, args=[src[None,:], dst[None,:], P0, P1], verbose=1)
    
    X_new = least_squares(ReprojectionError, X0.reshape(-1), args=[pts_src, pts_dst, P0, P1], diff_step= 0.001, verbose=2)
    X_new = X_new.x
        # X_final.append(X_new)

    X_final = X_new.reshape(len(X_new)//3, 3)

    return np.array(X_final)

