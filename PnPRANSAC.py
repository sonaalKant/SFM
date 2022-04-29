from LinearPnP import *
import numpy as np
from LinearTriangulation import make_P
from NonlinearTriangulation import *


def reprojectionError(X, x, P0):
    X = to_homogeneous(X)
    
    camera1 = (x[:,0] - P0[0,:]@X.T / (P0[2,:]@X.T))**2 + (x[:,1] - P0[1,:]@X.T / (P0[2,:]@X.T))**2
    
    return camera1

# def error(X, P, p1):
# 	X_ = np.hstack((X, np.ones((len(X), 1))))
# 	error = (p1[0] - ((P[0][0,:]@X_.T)/(P[0][2,:]@X_.T)))**2 + (p1[1] - ((P[0][1,:]@X_.T)/(P[0][2,:]@X_.T)))**2
# 	return error

# def PnPRANSAC(X, x, K):
# 	iterations = 2000
# 	RC = []
# 	inliers = []
# 	for i in range(iterations):
# 		S = 0
# 		idx = np.random.choice(len(X)-1, 8)
# 		X_ = X[idx]
# 		x_ = x[idx]
# 		R, C = LinearPnP(X_, x_, K)
# 		P = make_P(R, C)
# 		RC.append([R, C])
# 		for j in range(len(x)):
# 			e = error(X[j], P, x[j]) 
# 			if e < 0.1:
# 				S+=1
# 		inliers.append(S)
# 	inliers = np.array(inliers)
# 	idx = np.argmax(inliers)
# 	R, C = RC[idx]
# 	return [C[:,None], R]

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

