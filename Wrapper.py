import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
from Codes.utils import *
from Codes.GetInliersRANSAC import do_ransac
from Codes.EstimateFundamentalMatrix import getFundamentalMatrix
from Codes.EssentialMatrixFromFundamentalMatrix import getEssentialMatrix
from Codes.ExtractCameraPose import getCameraPose
from Codes.LinearTriangulation import do_triangulation
from Codes.DisambiguateCameraPose import do_chirality
from Codes.NonlinearTriangulation import nonlinear_triangulation
from Codes.PnP import *


def generate_data(data_path):
    matching_files = glob.glob(data_path + "/matching*.txt")
    data = OrderedDict()
    for m_file in matching_files:
        curr_image_id = m_file.split("/")[-1].split(".")[0].split("matching")[-1]
        data[curr_image_id] = OrderedDict()
        with open(m_file, "r") as f:
            lines = f.readlines()
            for l in lines[1:]:
                l = l.split()
                num, R,G,B, x,y = l[:6]
                for j in range(int(num)-1):
                    image_id, x_, y_ = l[6+3*j:9+3*j]

                    if image_id not in data[curr_image_id]:
                        data[curr_image_id][image_id] = {"src" : list(), "dst" : list()}
                    data[curr_image_id][image_id]["src"] += [[eval(x),eval(y)]]
                    data[curr_image_id][image_id]["dst"] += [[eval(x_),eval(y_)]]
        
    return data

def get_callibration(data_path):
    K = np.array([[  568.996140852, 0, 643.21055941],    
         [  0, 568.988362396, 477.982801038],
         [  0, 0, 1,]])
    return K

def drawMatches(img1, img2, k1, k2):
    new = np.concatenate((img1,img2),axis=1)
    k1 = np.array(k1).astype(int)
    k2 = np.array(k2).astype(int)
    k2[:,0] = k2[:,0] + img2.shape[1]
    for i in range(len(k1)):
    # import pdb; pdb.set_trace()
        cv2.line(new, (k1[i,0],k1[i,1]), (k2[i,0],k2[i,1]), [0,255,255], 1)
        cv2.circle(new, (k1[i,0],k1[i,1]), 10, [0,0,255])
        cv2.drawMarker(new, (k2[i,0],k2[i,1]), [0,255,0], markerSize=15)
    return new

def plot_points(X):
    colors = cm.rainbow(np.linspace(0, 1, len(X)))
    for points, c in zip(X, colors):
        points = np.array(points)
        x = points[:,0]
        y = points[:,2]
        plt.scatter(x, y, color=c)
    plt.grid(True)
    plt.show()

def main(data_path):
    data_dict = generate_data(data_path)
    
    K = get_callibration(data_path)

    data_dict = do_ransac(data_dict)

    images = list(data_dict.keys())
    images = sorted(images)
    im_id1, im_id2 = images[:2]

    F = data_dict[im_id1][im_id2]["F"]

    im1 = cv2.imread(data_path + f"/{im_id1}.jpg")
    im2 = cv2.imread(data_path + f"/{im_id2}.jpg")
    new = drawMatches(im1, im2, data_dict[im_id1][im_id2]["src"], data_dict[im_id1][im_id2]["dst"])

    cv2.imwrite("check.jpg", new)
    # import pdb;pdb.set_trace()

    E = getEssentialMatrix(F, K)

    C_list, R_list = getCameraPose(E)


    X_set = [do_triangulation(K, np.zeros((3,1)), np.eye(3), C, R, data_dict[im_id1][im_id2]["src"], data_dict[im_id1][im_id2]["dst"])
                for C,R in zip(C_list, R_list)]
    
    # plot_points(X_set)
    
    C,R,X,idxs = do_chirality(C_list, R_list, X_set)

    plot_points([X])

    src = data_dict[im_id1][im_id2]["src"][idxs]
    dst = data_dict[im_id1][im_id2]["dst"][idxs]

    C0 = np.zeros((3,1))
    R0 = np.eye(3)

    X_new = nonlinear_triangulation(K, C0, R0, C, R, src, dst, X)
    # X_new = nonlinear_triangulation2(K, C0, R0, C, R, src, dst, X)

    plot_points([X, X_new])

    C_list = [C0, C]
    R_list = [R0, R]

    registered_images = [im_id1, im_id2]
    registered_image_points = [src, dst]

    for i in range(2, len(images)):
        im_id2 = images[i]

        X_i, x = get_common_world_points(X, data_dict, registered_images, registered_image_points, im_id2)
        
        C_i, R_i = PnPRansac(X_i, x, K)
        C_i, R_i = NonLinearPnP(X_i, x, K, C_i, R_i)


        for j in range(i):
            im_id1 = images[j]
            C = C_list[j]
            R = R_list[j]

            X_new = do_triangulation(K, C,R, C_i, R_i, data_dict[im_id1][im_id2]["src"], data_dict[im_id1][im_id2]["dst"])
            X_new = np.array(X_new)
            X_new = nonlinear_triangulation(K, C,R, C_i, R_i, data_dict[im_id1][im_id2]["src"], data_dict[im_id1][im_id2]["dst"], X_new)

            X = X + X_new
        
        registered_images.append(im_id2)
        C_list.append(C_i)
        R_list.append(R_i)

        import pdb;pdb.set_trace()







    
    # plot_points([X])

    




if __name__ == '__main__':
    data_path = "../Data"
    main(data_path)