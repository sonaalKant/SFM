import numpy as np

def make_A(pt2, pt1):
	# import pdb; pdb.set_trace()
	for i in range(len(pt1)):
		x1, y1, z1 = pt1[i]
		x2, y2, z2 = pt2[i]
		if i == 0:
			A = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
		else:
			A = np.vstack((A, np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])))
	return A

def normalizePoints(pts):
	pts = np.array(pts)
	# import pdb; pdb.set_trace()
	pts_mean = np.mean(pts, axis=0)
	x_bar = pts_mean[0]
	y_bar = pts_mean[1]

	# origin of the new coordinate system should be located at the centroid of the image points
	x_s, y_s = pts[:,0] - x_bar, pts[:, 1] - y_bar

	# scale by the scaling factor
	s = ((2*len(pts))/np.mean(x_s**2 + y_s**2))**(0.5)

	# construct transformation matrix (translation+scaling)
	T_S = np.diag([s,s,1])
	T_T = np.array([[1, 0, -x_bar],[0, 1, -y_bar],[0, 0, 1]])
	Ta = np.dot(T_S, T_T)

	x = np.column_stack((pts, np.ones(pts.shape[0])))
	x_norm = (Ta@x.T).T
	return x_norm, Ta

def get_F(pt1, pt2):
	pt1 = np.array(pt1)
	pt2 = np.array(pt2)
	pt1 = np.column_stack((pt1, np.ones(pt1.shape[0])))
	pt2 = np.column_stack((pt2, np.ones(pt2.shape[0])))
	# pt1, T1 = normalizePoints(pt1)
	# pt2, T2 = normalizePoints(pt2)

	A = make_A(pt1, pt2)
	u,s,v = np.linalg.svd(A)
	F = v[len(v)-1,:].reshape((3,3))
	U, S, V = np.linalg.svd(F)
	S = np.diag(S)
	S[2,2] = 0
	F = U@S@V
	# F = T2.T@(F@T1)

	return F