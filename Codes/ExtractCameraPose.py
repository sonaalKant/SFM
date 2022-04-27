import numpy as np
import cv2

def getCameraPose(E):
    U,S,V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    C = U[:,2].reshape(-1,1)

    R1 = U @ (W @ V)
    R1, C1 = (R1, C) if np.linalg.det(R1) > 0 else (-R1, -C)

    R3 = U @ (W.T @ V)
    R3, C3 = (R3,C) if np.linalg.det(R3) > 0 else (-R3,-C)

    return [C1, -C1, C3, -C3], [R1, R1, R3, R3]