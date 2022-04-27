import numpy as np
import cv2

def do_triangulation(K, C1, R1, C2, R2, x1, x2):
    I = np.eye(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    X = list()
    for p1, p2 in zip(x1, x2):
        A = list()
        u1,v1 = p1
        u2,v2 = p2

        A.append([ v1*P1[2,0] - P1[1,0], v1*P1[2,1] - P1[1,1], v1*P1[2,2] - P1[1,2], v1*P1[2,3] - P1[1,3] ])
        A.append([ u1*P1[2,0] - P1[0,0], u1*P1[2,1] - P1[0,1], u1*P1[2,2] - P1[0,2], u1*P1[2,3] - P1[0,3] ])

        A.append([ v2*P2[2,0] - P2[1,0], v2*P2[2,1] - P2[1,1], v2*P2[2,2] - P2[1,2], v2*P2[2,3] - P2[1,3] ])
        A.append([ u2*P2[2,0] - P2[0,0], u2*P2[2,1] - P2[0,1], u2*P2[2,2] - P2[0,2], u2*P2[2,3] - P2[0,3] ])

        A = np.array(A)
        U,S,V = np.linalg.svd(A)
        idx = np.argmin(S)
        x = V.T[:, idx]
        x = x / x[3]
        X.append(x[:3])
    
    return X

    


    