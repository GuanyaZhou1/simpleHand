import json
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
import albumentations as A
from typing import List, Dict
from itertools import cycle
from cfg import _CONFIG
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transforms import GetRandomScaleRotation, MeshAffine, RandomHorizontalFlip, \
            get_points_center_scale, RandomChannelNoise, BBoxCenterJitter, MeshPerspectiveTransform
import random

DATA_CFG = _CONFIG["DATA"]
IMAGE_SHAPE: List = DATA_CFG["IMAGE_SHAPE"][:2]
NORMALIZE_3D_GT = DATA_CFG['NORMALIZE_3D_GT']
AUG_CFG: Dict = DATA_CFG["AUG"]
ROOT_INDEX = DATA_CFG['ROOT_INDEX']

def read_info(img_path):
    info_path = img_path.replace('.jpg', '.json')
    with open(info_path) as f:
        info = json.load(f)
    return info

with open(DATA_CFG['JSON_DIR']) as f:
    all_image_info = json.load(f)
all_info = []
for image_path in tqdm(all_image_info):
    info = read_info(image_path)
    info['image_path'] = image_path
    all_info.append(info)

# use training_xyz 拟合传感器数据信号
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

class HandDataset(Dataset):
    def __init__(self, all_info):
        super().__init__()

        self.init_aug_funcs()
        self.all_info = all_info

    def __len__(self):
        return len(self.all_info)
    
    def init_aug_funcs(self):
        self.random_channel_noise = RandomChannelNoise(**AUG_CFG['RandomChannelNoise'])
        self.random_bright = A.RandomBrightnessContrast(**AUG_CFG["RandomBrightnessContrastMap"])            
        self.random_flip = RandomHorizontalFlip(**AUG_CFG["RandomHorizontalFlip"])
        self.bbox_center_jitter = BBoxCenterJitter(**AUG_CFG["BBoxCenterJitter"])
        self.get_random_scale_rotation = GetRandomScaleRotation(**AUG_CFG["GetRandomScaleRotation"])
        self.mesh_affine = MeshAffine(IMAGE_SHAPE[0])
        self.mesh_perspective_trans = MeshPerspectiveTransform(IMAGE_SHAPE[0])
        
        self.root_index = ROOT_INDEX

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
    
    def __getitem__(self, index):
        data_info = self.all_info[index]
        img = self.read_image(data_info['image_path'])
        # keypoints2d = np.array(data_info['uv'], dtype=np.float32)
        keypoints3d = np.array(data_info['xyz'], dtype=np.float32)
        K = np.array(data_info['K'], dtype=np.float32)
        
        proj_points = (K @ keypoints3d.T).T
        keypoints2d = proj_points[:, :2] / (proj_points[:, 2:] + 1e-7)
        
        vertices = np.array(data_info['vertices']).astype('float32')

        h, w = img.shape[:2]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            

        uv_norm = keypoints2d.copy()
        uv_norm[:, 0] /= w   
        uv_norm[:, 1] /= h

        coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
        coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

        valid_points = [keypoints2d[i] for i in range(len(keypoints2d)) if coord_valid[i]==1]
        
        points = np.array(valid_points)
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        center = (max_coord + min_coord)/2
        scale = max_coord - min_coord
                
        results = {
            "img": img,
            "keypoints2d": keypoints2d,
            "keypoints3d": keypoints3d,
            "vertices": vertices,
            
            "center": center,
            "scale": scale,
            "K": K,
        }
        # 随机 0.1 概率 mask 手
        if random.random() < 0.1:
            results['img'] = self.mask_hand(results['img'], results['keypoints2d'])
        # 1. Crop and Rot
        results = self.bbox_center_jitter(results)
        results = self.get_random_scale_rotation(results)
        # results = self.mesh_affine(results)
        results = self.mesh_perspective_trans(results)

        # 2. 3D KP Root Relative
        root_point = results['keypoints3d'][self.root_index].copy()
        results['keypoints3d'] = results['keypoints3d'] - root_point[None, :]
        results['vertices'] = results['vertices'] - root_point[None, :]
        
        hand_img_len = IMAGE_SHAPE[0]
        root_depth = root_point[2]

        hand_world_len = 0.2
        fx = results['K'][0][0]
        fy = results['K'][1][1]
        camare_relative_k = np.sqrt(fx * fy * (hand_world_len**2) / (hand_img_len**2))
        gamma = root_depth / camare_relative_k
        # 3. Random Flip 
        results = self.random_flip(results)
        # 4. Image aug
        results = self.random_channel_noise(results)
        results['img'] = self.random_bright(image=results['img'])['image']

        trans_uv = results["keypoints2d"]
        trans_uv[:, 0] /= IMAGE_SHAPE[0]
        trans_uv[:, 1] /= IMAGE_SHAPE[1]

        trans_coord_valid = (trans_uv > 0).astype("float32") * (trans_uv < 1).astype("float32") # Nx2x21x2
        trans_coord_valid = trans_coord_valid[:, 0] * trans_coord_valid[:, 1]
        trans_coord_valid *= coord_valid

        xyz = results["keypoints3d"]
        if NORMALIZE_3D_GT:
            joints_bone_len = np.sqrt(((xyz[0:1] - xyz[9:10])**2).sum(axis=-1, keepdims=True) + 1e-8)
            xyz = xyz  / joints_bone_len
        
        xyz_valid = 1

        if trans_coord_valid[9] == 0 and trans_coord_valid[0] == 0:
            xyz_valid = 0

        img = results['img']
        img = np.transpose(img, (2,0,1))
        fit_result = fit_sensor_data(xyz.flatten())
        fit_result = np.array(fit_result, dtype=np.float32)
        data = {
            "img": img,
            "uv": results["keypoints2d"],
            "xyz": xyz,
            "vertices": results['vertices'],                
            "uv_valid": trans_coord_valid,
            "gamma": gamma,
            "xyz_valid": xyz_valid,
            "fit_sensor_data": fit_result
        }

        return data

def build_train_loader(batch_size):
	dataset = HandDataset(all_info)
	sampler = RandomSampler(dataset, replacement=True)
	dataloader = (DataLoader(dataset, batch_size=batch_size, sampler=sampler))
	return iter(dataloader)

# if __name__ == "__main__":
#     train_loader = build_train_loader(_CONFIG['TRAIN']['DATALOADER']['MINIBATCH_SIZE_PER_DIVICE'])
#     batch = next(train_loader)
#     with open('batch_data.pkl', 'rb') as f:
#         pickle.dump(batch, f)
#     from IPython import embed 
#     embed()
#     exit()