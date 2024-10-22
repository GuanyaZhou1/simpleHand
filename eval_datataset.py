import numpy as np
import json
from functools import lru_cache
import cv2
import pickle
from tqdm import tqdm
from typing import List, Dict

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler


from kp_preprocess import get_2d3d_perspective_transform, get_points_bbox, get_points_center_scale

import math
def calculate_euler_angles(result):
    x0 = result[0]
    y0 = -result[2]
    z0 = result[1]
    
    x5 = result[15]
    y5 = -result[17]
    z5 = result[16]

    x9 = result[27]
    y9 = -result[29]
    z9 = result[28]

    x13 = result[39]
    y13 = -result[41]
    z13 = result[40]

    x17 = result[51]
    y17 = -result[53]
    z17 = result[52]
    #计算向量
    v09 = np.array([x9-x0, y9-y0, z9-z0])
    v05 = np.array([x5-x0, y5-y0, z5-z0])
    v013 = np.array([x13-x0, y13-y0, z13-z0])
    v017 = np.array([x17-x0, y17-y0, z17-z0])
    v517 = np.array([x17-x5, y17-y5, z17-z5])
    
    # 进行叉乘，得到手掌平面法向量
    b = np.cross(v05,v09)
    b1 = np.cross(v09,v013)
    b2 = np.cross(v013,v017)
    b3 = np.cross(v05,v013)
    b4 = np.cross(v05,v017)
    b5 = np.cross(v09,v017)

    b_mean = (b+b1+b2+b3+b4+b5)/6
    z1 = b_mean/np.linalg.norm(b_mean)
    y1 = v09/np.linalg.norm(v09)
    a = np.cross(z1,y1)
    x1 = a/np.linalg.norm(a)
    

    yaw = math.atan2(-y1[2],-z1[2])
    pitch = math.asin(x1[2])
    roll = math.atan2(x1[1],x1[0])

    yaw_deg = math.degrees(yaw)/180.0
    pitch_deg = math.degrees(pitch)/180.0
    roll_deg = math.degrees(roll)/180.0
    return [yaw_deg, pitch_deg, roll_deg]


def calculate_finger_angels(result):
    final_result = []
    # 1,5,9...17是每个手指的开始
    for i in [1,5,9,13,17]:
        joint0 = np.array(result[0:3]) 
        joint1 = np.array(result[i*3:(i+1)*3])
        joint2 = np.array(result[(i+1)*3:(i+2)*3])
        joint3 = np.array(result[(i+2)*3:(i+3)*3])
        joint4 = np.array(result[(i+3)*3:(i+4)*3])
        # get the vector of neighboring joints by subtracting their coordinates
        vector01 = joint1 - joint0
        vector12 = joint2 - joint1
        vector23 = joint3 - joint2
        vector34 = joint4 - joint3

        
        # calculate the angle from the neighboring vectors using arccos of their dot product divided by their norms
        angle01_12 = math.acos(np.dot(vector01,vector12)/(np.linalg.norm(vector01)*np.linalg.norm(vector12)))
        angle12_23 = math.acos(np.dot(vector12,vector23)/(np.linalg.norm(vector12)*np.linalg.norm(vector23)))
        angle23_34 = math.acos(np.dot(vector23,vector34)/(np.linalg.norm(vector23)*np.linalg.norm(vector34)))

        # convert the angles from radians to degrees
        angle01_12 = math.degrees(angle01_12)/180.0
        angle12_23 = math.degrees(angle12_23)/180.0
        angle23_34 = math.degrees(angle23_34)/180.0

        # average the angles to represent the degree of curvature of the finger
        if i==1:
            curvature =  ( angle12_23 + angle23_34)/2
        else:
            curvature = (angle01_12 + angle12_23 )/2

        # add the curvature to the final result list
        final_result.append(curvature)
    return final_result
def fit_sensor_data(result):
    euler_angles = calculate_euler_angles(result)
    finger_angles = calculate_finger_angels(result)
    return euler_angles + finger_angles

class HandMeshEvalDataset(Dataset):
    def __init__(self, json_path, img_size=(224, 224), scale_enlarge=1.2, rot_angle=0):
        super().__init__()

        with open(json_path) as f:
            self.all_image_info = json.load(f)
        self.all_info = [{"image_path": image_path} for image_path in self.all_image_info]
        self.img_size = img_size
        self.scale_enlarge = scale_enlarge
        self.rot_angle = rot_angle

    def __len__(self):
        return len(self.all_image_info)
    
    def read_image(self, img_path):
        img = cv2.imread(img_path)
        return img

    def mask_hand(self,img,keypoints2d):
        h,w = img.shape[:2]
        mask = np.zeros((h,w),dtype=np.uint8)
        points = keypoints2d.astype(np.int32)
        x_min,y_min = points.min(axis=0)
        x_max,y_max = points.max(axis=0)
        cv2.rectangle(mask,(x_min,y_min),(x_max,y_max),0,-1)

        img = cv2.bitwise_and(img,img,mask=mask)
        return img

    def read_info(self, img_path):
        info_path = img_path.replace('.jpg', '.json')
        with open(info_path) as f:
            info = json.load(f)
        return info
    
    def __getitem__(self, index):
        image_path = self.all_image_info[index]
        img = self.read_image(image_path)
        ori_img = img.copy()
        data_dict = self.read_info(image_path)
        h, w = img.shape[:2]
        K = np.array(data_dict['K'])
        if "uv" in data_dict:
            uv = np.array(data_dict['uv'])
            img = self.mask_hand(img,uv)
            xyz = np.array(data_dict['xyz'])
            vertices = np.array(data_dict['vertices'])
            uv_norm = uv.copy()
            uv_norm[:, 0] /= w   
            uv_norm[:, 1] /= h

            coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
            coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

            valid_points = [uv[i] for i in range(len(uv)) if coord_valid[i]==1]        
            if len(valid_points) <= 1:
                valid_points = uv

            points = np.array(valid_points)
            min_coord = points.min(axis=0)
            max_coord = points.max(axis=0)
            center = (max_coord + min_coord)/2
            scale = max_coord - min_coord
        else:
            bbox = data_dict['bbox']
            x1, y1, x2, y2 = bbox[:4]
            center = np.array([(x1 + x2)/2, (y1 + y2) / 2])
            scale = np.array([x2 - x1, y2- y1])
            uv = np.zeros((21, 2), dtype=np.float32)
            xyz = np.zeros((21, 3), dtype=np.float32)
        
        ori_xyz = xyz.copy()
        fit_result = fit_sensor_data(xyz.flatten())
        fit_result = np.array(fit_result, dtype=np.float32)

        ori_vertices = vertices.copy()
        scale = scale * self.scale_enlarge
        # perspective trans
        new_K, trans_matrix_2d, trans_matrix_3d = get_2d3d_perspective_transform(K, center, scale, self.rot_angle, self.img_size[0])
        img_processed = cv2.warpPerspective(img, trans_matrix_2d, self.img_size)
        ori_img_processed = cv2.warpPerspective(ori_img,trans_matrix_2d,self.img_size)
        new_uv = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        new_uv = (trans_matrix_2d @ new_uv.T).T
        new_uv = new_uv[:, :2] / new_uv[:, 2:]
        new_xyz = (trans_matrix_3d @ xyz.T).T       
        
        vertices = trans_matrix_3d.dot(vertices.T).T

         

        if img_processed.ndim == 2:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)
        img_processed = np.transpose(img_processed, (2, 0, 1))
        if ori_img_processed.ndim == 2:
            ori_img_processed = cv2.cvtColor(ori_img_processed, cv2.COLOR_GRAY2BGR)
        ori_img_processed = np.transpose(ori_img_processed, (2, 0, 1))
        return {
            "img": np.ascontiguousarray(img_processed),
            "trans_matrix_2d": trans_matrix_2d,
            "trans_matrix_3d": trans_matrix_3d,            
            "K": new_K,
            "uv": new_uv,
            "xyz": new_xyz,
            "vertices": vertices,            
            "scale": self.img_size[0],
            "ori_xyz":ori_xyz,
            "ori_vertices":ori_vertices,
            "fit_sensor_data":fit_result,
            "ori_img":np.ascontiguousarray(ori_img_processed)

        }
        
    def __str__(self):
        return json.dumps(len(self.all_image_info))
