import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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