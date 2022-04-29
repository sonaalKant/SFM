import numpy as np
import random
from Codes.utils import *
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares, curve_fit

# Refer: https://www.cis.upenn.edu/~cis580/Spring2015/Lectures/cis580-13-LeastSq-PnP.pdf

def reprojectionError(X, x, P0):
    X = to_homogeneous(X)
    
    camera1 = (x[:,0] - P0[0,:]@X.T / (P0[2,:]@X.T))**2 + (x[:,1] - P0[1,:]@X.T / (P0[2,:]@X.T))**2
    
    return camera1

def LinearPnP(K,x,X):
    x = to_homogeneous(x)
    X = to_homogeneous(X)

    K_inv = np.linalg.inv(K)
    x = (K_inv@x.T).T

    zeros = np.zeros((4))
    A = list()
    for i in range(len(x)):
        u,v,_ = x[i]
        a = np.vstack( (np.hstack((zeros, -X[i], v*X[i])), np.hstack((X[i], zeros, -u*X[i])), np.hstack((-v*X[i], u*X[i], zeros)) ))
        A.append(a)
    
    A = np.array(A).reshape(-1,12)
    U,S,V = np.linalg.svd(A)
    P = V[-1].reshape(3,4)
    R = P[:,:3]
    U,S,V = np.linalg.svd(R)
    R = U@V
    t = P[:,3]
    C = -R@t

    return R,C


def PnPRansac(X,x,K):
    M = 2000
    pnp_best = None
    idxs_best = list()
    eps = 5

    for i in range(M):
        idxs = random.sample( list(np.arange(len(x))), 6)

        ps = x[idxs]
        pd = X[idxs]

        R,C = LinearPnP(K, ps, pd)
        
        I = np.eye(3)
        P = np.dot(K, np.dot(R, np.hstack((I, -C[:,None]))))
        
        dist = reprojectionError(X,x,P)


        S = np.where(abs(dist) < eps)[0]

        if len(S) > len(idxs_best):
            print(len(S))
            idxs_best = S
            pnp_best = [C[:,None],R]
    
    return pnp_best

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

