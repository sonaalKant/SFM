import numpy as np
import cv2

def getFundamentalMatrix(points1, points2):
    x1,y1 = points1[:,0], points1[:,1]
    x2,y2 = points2[:,0], points2[:,1]

    A = np.concatenate([(x1*x2)[:,np.newaxis], (x1*y2)[:,np.newaxis], x1[:,np.newaxis], 
                    (y1*x2)[:,np.newaxis], (y1*y2)[:,np.newaxis], y1[:,np.newaxis], 
                    x2[:,np.newaxis], y2[:,np.newaxis], np.ones((len(x1), 1))], -1)
    
    U,S,V = np.linalg.svd(A)
    F = V[:,-1].reshape(3,3)

    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ (np.diag(S) @ V)

    return F
