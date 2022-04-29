import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from GetInlierRANSAC import do_ransac
from EstimateFundamentalMatrix import get_F as getFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import get_E as getEssentialMatrix
from ExtractCameraPose import ExtractCameraPose as getCameraPose
from LinearTriangulation import LinearTriangulation as do_triangulation
from DisambiguateCameraPose import DisambiguateCameraPose as do_chirality
from NonlinearTriangulation import NonlinearTriangulation as nonlinear_triangulation
from PnPRANSAC import *
from NonlinearPnP import *
from utils import *

def generate_data(data_path):
    matching_files = glob.glob(data_path + "/matching*.txt")
    data = dict()
    for m_file in matching_files:
        curr_image_id = m_file.split("/")[-1].split(".")[0].split("matching")[-1]
        data[curr_image_id] = dict()
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
         [  0, 0, 1]])
    return K

def plot_points(X):
    colors = cm.rainbow(np.linspace(0, 1, len(X)))
    for points, c in zip(X, colors):
        points = np.array(points)
        x = points[:,0]
        y = points[:,2]
        plt.scatter(x, y, color=c)
    plt.grid(True)
    plt.show()

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

def main(data_path):
    data_dict = generate_data(data_path)
    
    K = get_callibration(data_path)

    # pts1 = np.array(data_dict['1']['2']['src'])
    # pts2 = np.array(data_dict['1']['2']['dst'])
    # # print(pts1.shape)
    # img1 = cv2.imread(data_path+'/1.jpg')
    # img2 = cv2.imread(data_path+'/2.jpg')
    # import pdb; pdb.set_trace()

    data_dict = do_ransac(data_dict)
    images = list(data_dict.keys())
    images = sorted(images)
    im_id1, im_id2 = images[:2]

    F = data_dict[im_id1][im_id2]["F"]
    # print(pts1.shape)
    # new = drawMatches(img1, img2, pts1, pts2)
    # cv2.imwrite('ransac.png', new)
    
    # cv2.waitKey(0)
    # import pdb; pdb.set_trace()
    


    E = getEssentialMatrix(F, K)
    print('F:\n',F, '\nE:\n',E)

    R_list, C_list = getCameraPose(E)


    X_set = [do_triangulation(K, C, R, pts1, pts2)
                for C,R in zip(C_list, R_list)]

    for k in X_set:
        plt.scatter(k[:,0], k[:,2], s=0.2)
    plt.show()
        
    R,C,X,idxs, idx = do_chirality(C_list, R_list, X_set)
    src = pts1[idxs]
    dst = pts2[idxs]
    X = X[idxs]
    plt.scatter(X[:,0], X[:,2], s=0.4)
    plt.show()
    # print(X)
    # import pdb;pdb.set_trace()
    # exit()

    C0 = np.zeros((3,1))
    R0 = np.eye(3)
    X_new = nonlinear_triangulation(R0, C0, R, C, X, pts1, pts2)
    plt.scatter(X_new[:,0], X_new[:,2], s=0.4)
    plt.scatter(X_set[idx][:,0], X_set[idx][:,2], s=0.4)
    plt.show()

    C_list = [C0, C]
    R_list = [R0, R]

    registered_images = [im_id1, im_id2]
    registered_image_points = [src, dst]

    
    # new_R, new_C = LinearPnP(X_new, pts1, K)

    import pdb;pdb.set_trace()

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


    




if __name__ == '__main__':
    data_path = "./Data"
    main(data_path)