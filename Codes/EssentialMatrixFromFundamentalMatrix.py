import numpy as np
import cv2

def getEssentialMatrix(F, K):
    E = K.T @ (F @ K)
    U,S,V = np.linalg.svd(E)
    S = np.array([1,1,0])
    E = U @ (np.diag(S) @ V)
    return E
