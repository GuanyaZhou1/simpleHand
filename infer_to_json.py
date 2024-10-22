
import json
import numpy as np
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from cfg import _CONFIG
from hand_net import HandNet
from eval_datataset import HandMeshEvalDataset
from utils import get_log_model_dir
from torch import Tensor
from scipy.linalg import orthogonal_procrustes
import open3d as o3d
import numpy as np


class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
def verts2pcd(verts, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    if color is not None:
        if color == 'r':
            pcd.paint_uniform_color([1, 0.0, 0])
        if color == 'g':
            pcd.paint_uniform_color([0, 1.0, 0])
        if color == 'b':
            pcd.paint_uniform_color([0, 0, 1.0])
    return pcd


def calculate_fscore(gt, pr, th=0.01):
    gt = verts2pcd(gt)
    pr = verts2pcd(pr)
    # d1 = o3d.compute_point_cloud_to_point_cloud_distance(gt, pr) # closest dist for each gt point
    # d2 = o3d.compute_point_cloud_to_point_cloud_distance(pr, gt) # closest dist for each pred point
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall


def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
joint_connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],  # 食指
    [0, 9], [9, 10], [10, 11], [11, 12],  # 中指
    [0, 13], [13, 14], [14, 15], [15, 16],  # 无名指
    [0, 17], [17, 18], [18, 19], [19, 20],  # 小指
    [5, 9], [9, 13], [13, 17]  # 手掌
]

def visualize_hand(joints, vertices, original_image, save_path, show_mesh=True):
    """
    可视化手部姿态（关节）和网格，并与原始图像拼合后保存。
    
    :param joints: 手部关节的3D坐标 (21, 3)
    :param vertices: 手部网格的3D顶点 (778, 3)
    :param original_image: 原始输入图像 (H, W, C) numpy数组
    :param save_path: 保存图像的路径
    :param show_mesh: 是否展示手部网格
    """
    fig = plt.figure(figsize=(12, 6))

    # 子图1：原始图像
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 子图2：手部姿态和网格
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # 绘制手部关节
    joints = np.array(joints)
    ax2.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', label='Joints', s=30)
    # 绘制关节之间的连线
    for connection in joint_connections:
        joint_start = joints[connection[0]]
        joint_end = joints[connection[1]]
        ax2.plot([joint_start[0], joint_end[0]],
                 [joint_start[1], joint_end[1]],
                 [joint_start[2], joint_end[2]], c='k')


    # 绘制手部网格
    if show_mesh:
        vertices = np.array(vertices)
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', label='Vertices', s=1)
    
    # 设置3D图像的标签
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Hand Pose and Mesh Visualization')
    ax2.legend()

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def infer_single_json(val_cfg, bmk, model, rot_angle=0):
    dataset = HandMeshEvalDataset(bmk["json_dir"], val_cfg["IMAGE_SHAPE"], bmk["scale_enlarge"], rot_angle=rot_angle)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=val_cfg["BATCH_SIZE"], num_workers=4, timeout=60)
    # 定义图像转换，将Tensor转换为PIL图像
    to_pil = transforms.ToPILImage()

    HAND_WORLD_LEN = 0.2
    ROOT_INDEX = _CONFIG['DATA'].get('ROOT_INDEX', 9)
        
    pred_uv_list = []
    pred_joints_list = []
    pred_vertices_list = []
    gt_joints_list = []
    gt_vertices_list = []



    for cur_iter, batch_data in enumerate(tqdm(dataloader)):
        for k in batch_data:
            batch_data[k] = batch_data[k].cuda().float()
        image = batch_data['img']
        ori_img = batch_data['ori_img']
        sensor_data = Tensor(batch_data['fit_sensor_data']).cuda().float()
        scale = batch_data['scale']
        K = batch_data['K']
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        dx = K[:, 0, 2]
        dy = K[:, 1, 2]
                
        trans_matrix_2d = batch_data['trans_matrix_2d']
        trans_matrix_3d = batch_data['trans_matrix_3d']
                
        
        trans_matrix_2d_inv = torch.linalg.inv(trans_matrix_2d)        
        trans_matrix_3d_inv = torch.linalg.inv(trans_matrix_3d)
        
        with torch.no_grad():
            res = model(image,sensor_data)
            joints = res["joints"]
            uv = res["uv"]
            vertices = res['vertices']
            gt_uv = batch_data
            
        vertices = vertices.reshape(-1, 778, 3)
        joints = joints.reshape(-1, 21, 3)
        uv = uv.reshape(-1, 21, 2) * val_cfg['IMAGE_SHAPE'][0]

        
        joints_root = joints[:, ROOT_INDEX][:, None, :]
        joints = joints - joints_root
        vertices = vertices - joints_root
                
        joints = (trans_matrix_3d_inv @ torch.transpose(joints, 1, 2)).transpose(1, 2) 
        vertices = (trans_matrix_3d_inv @ torch.transpose(vertices, 1, 2)).transpose(1, 2) 
        
        b, j = uv.shape[:2]
        pad = torch.ones((b, j, 1)).to(uv.device)
        uv = torch.concat([uv, pad], dim=2)        
        uv = (trans_matrix_2d_inv @ torch.transpose(uv, 1, 2)).transpose(1, 2)
        uv = uv[:, :, :2] / (uv[:, :, 2:] + 1e-7)

        pred_uv_list += uv.cpu().numpy().tolist()
        pred_joints_list += joints.cpu().numpy().tolist()
        pred_vertices_list += vertices.cpu().numpy().tolist()
        gt_joints_list += batch_data['xyz'].cpu().numpy().tolist()
        gt_vertices_list += batch_data['vertices'].cpu().numpy().tolist()
        # 提取并转换原始图像
        # 假设图像为 [B, C, H, W] 格式
        img_np = image[0].cpu().numpy() 
        ori_img_np = ori_img[0].cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # 转换为 [H, W, C]
        ori_img_np = np.transpose(ori_img_np, (1, 2, 0))  # 转换为 [H, W, C]
        img_np = (img_np * 255).astype(np.uint8)  # 假设图像已经被归一化到 [0,1]
        ori_img_np = (ori_img_np * 255).astype(np.uint8)  # 假设图像已经被归一化到 [0,1]
        output_dir = "/gpfs/public/vl/yzg/proj/simpleHand/eval_show"
        os.makedirs(output_dir, exist_ok=True)
        # 调用可视化函数并保存图像
        save_filename = f"visualization_iter_{cur_iter}_sample_0.png"
        save_path = os.path.join(output_dir, save_filename)
        visualize_hand(joints[0].cpu().numpy(), vertices[0].cpu().numpy(), ori_img_np, save_path, show_mesh=True)
    return pred_uv_list, pred_joints_list, pred_vertices_list, gt_joints_list, gt_vertices_list




def main(epoch, tta=False, postfix=""):

    val_cfg = _CONFIG['VAL']
    
    assert epoch.startswith('epoch'), "type epoch_15 for the 15th epoch"
    log_model_dir = get_log_model_dir(_CONFIG['NAME'])
    model_path = os.path.join(log_model_dir, epoch)
    # from IPython import embed 
    # embed()
    # exit()
    print(model_path)
    model = HandNet(_CONFIG, pretrained=False)

    checkpoint = torch.load(open(model_path, "rb"), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.cuda()
    
    bmk = val_cfg['BMK']
    
    dataset = HandMeshEvalDataset(bmk["json_dir"], val_cfg["IMAGE_SHAPE"], bmk["scale_enlarge"])

    pred_uv_list, xyz_pred_list, verts_pred_list, xyz_gt_list, verts_gt_list = infer_single_json(val_cfg, bmk, model, rot_angle=0)

    result_json_path = os.path.join(log_model_dir, "evals", bmk['name'], f"{epoch}{postfix}.json")
    
    for pred_uv, pred_xyz, pred_vertices, gt_joints, gt_vertices, ori_info in zip(pred_uv_list, xyz_pred_list, verts_pred_list, xyz_gt_list, verts_gt_list, dataset.all_info):
        ori_info['pred_uv'] = pred_uv
        ori_info['pred_xyz'] = pred_xyz
        ori_info['pred_vertices'] = pred_vertices
        ori_info['xyz'] = gt_joints
        ori_info['vertices'] = gt_vertices

    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score, f_score_aligned = list(), list()
    f_threshs = [0.005, 0.015]
    shape_is_mano = None
    for idx in range(len(xyz_gt_list)):
        xyz, verts = xyz_gt_list[idx], verts_gt_list[idx]
        xyz, verts = [np.array(x) for x in [xyz, verts]]

        xyz_pred, verts_pred = xyz_pred_list[idx], verts_pred_list[idx]
        xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )

        if shape_is_mano is None:
            if verts_pred.shape[0] == verts.shape[0]:
                shape_is_mano = True
            else:
                shape_is_mano = False

        if shape_is_mano:
            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred
            )

        # align predictions
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        if shape_is_mano:
            verts_pred_aligned = align_w_scale(verts, verts_pred)
        else:
            # use trafo estimated from keypoints
            trafo = align_w_scale(xyz, xyz_pred, return_trafo=True)
            verts_pred_aligned = align_by_trafo(verts_pred, trafo)

        # Aligned errors
        eval_xyz_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_aligned
        )

        if shape_is_mano:
            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned
            )

        # F-scores
        l, la = list(), list()
        for t in f_threshs:
            # for each threshold calculate the f score and the f score of the aligned vertices
            f, _, _ = calculate_fscore(verts, verts_pred, t)
            l.append(f)
            f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
            la.append(f)
        f_score.append(l)
        f_score_aligned.append(la)

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))

    if shape_is_mano:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))
    else:
        mesh_mean3d, mesh_auc3d, mesh_al_mean3d, mesh_al_auc3d = -1.0, -1.0, -1.0, -1.0

        pck_mesh, thresh_mesh = np.array([-1.0, -1.0]), np.array([0.0, 1.0])
        pck_mesh_al, thresh_mesh_al = np.array([-1.0, -1.0]), np.array([0.0, 1.0])

    print('F-scores')
    f_out = list()
    f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
    for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
        print('F@%.1fmm = %.4f' % (t*1000, f.mean()), '\tF_aligned@%.1fmm = %.4f' % (t*1000, fa.mean()))
        f_out.append('f_score_%d: %f' % (round(t*1000), f.mean()))
        f_out.append('f_al_score_%d: %f' % (round(t*1000), fa.mean()))


    # with os.open(result_json_path, 'w') as f:
    #     json.dump(dataset.all_info, f)
            
    #     print(f"Result save to {result_json_path}")
        

if __name__ == "__main__":
    from fire import Fire
    Fire(main)

