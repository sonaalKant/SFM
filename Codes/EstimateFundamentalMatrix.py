import numpy as np
import cv2

def normalizePoints(pts):
	pts = np.array(pts)
	# import pdb; pdb.set_trace()
	pts_mean = np.mean(pts, axis=0)
	x_bar = pts_mean[0]
	y_bar = pts_mean[1]

	# origin of the new coordinate system should be located at the centroid of the image points
	x_s, y_s = pts[:,0] - x_bar, pts[:, 1] - y_bar

	# scale by the scaling factor
	s = (2/np.mean(x_s**2 + y_s**2))**(0.5)

	# construct transformation matrix (translation+scaling)
	T_S = np.diag([s,s,1])
	T_T = np.array([[1, 0, -x_bar],[0, 1, -y_bar],[0, 0, 1]])
	Ta = np.dot(T_S, T_T)

	x = np.column_stack((pts, np.ones(pts.shape[0])))
	x_norm = (Ta@x.T).T
	return x_norm, Ta

def getFundamentalMatrix(points1, points2):
    points1, T1 = normalizePoints(points1)
    points2, T2 = normalizePoints(points2)

    x1,y1 = points1[:,0], points1[:,1]
    x2,y2 = points2[:,0], points2[:,1]

    A = np.concatenate([(x1*x2)[:,np.newaxis], (x2*y1)[:,np.newaxis], x2[:,np.newaxis], 
                    (y2*x1)[:,np.newaxis], (y2*y1)[:,np.newaxis], y2[:,np.newaxis], 
                    x1[:,np.newaxis], y1[:,np.newaxis], np.ones((len(x1), 1))], -1)

    
    U,S,V = np.linalg.svd(A)
    F = V.T[:,-1].reshape(3,3)

    U,S,V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ (np.diag(S) @ V)
    F = T2.T@(F@T1)
    F = F / F[2,2]
    return F

