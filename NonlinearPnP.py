import numpy as np
from utils import *
from PnPRANSAC import *
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, curve_fit


def loss_func(params, X, x, K):
    q = params[:4]
    C = params[4:]
    R = Rotation.from_quat(q).as_matrix()

    I = np.eye(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C[:,None]))))

    loss = reprojectionError(X,x,P)
    return np.sum(loss)

def NonLinearPnP(X, x, K, C, R):
    
    r = Rotation.from_matrix(R)
    q = r.as_quat()

    C = C.reshape(-1)

    params = q.tolist() + C.tolist()

    new_params = least_squares(loss_func, params, args=[X,x,K], verbose=2)
    new_params = new_params.x

    q = new_params[:4]
    C = new_params[4:]

    R = Rotation.from_quat(q).as_matrix()


    return C[:,None],R