import open3d as o3d
import numpy as np
from metrics.metric import l1_cd, l2_cd, emd, f_score
from dcd.dcd import calc_dcd
import torch
import os
from tools.noise_reuction import del_zdm

def fidelity(pc_np, gt_np):
    pc = o3d.geometry.PointCloud()
    gt = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_np)
    gt.points = o3d.utility.Vector3dVector(gt_np)
    dist = pc.compute_point_cloud_distance(gt)
    dist = np.array(dist)
    return np.mean(dist)

def read_res_and_gt(res_path, gt_path):
    pcd_res = o3d.io.read_point_cloud(res_path)
    pcd_gt = o3d.io.read_point_cloud(gt_path)
    points_res = np.asarray(pcd_res.points)
    points_gt = np.asarray(pcd_gt.points)
    return points_res, points_gt

def np2tensor(points):
    points = torch.from_numpy(points)
    return points.to(torch.device("cuda:3")).unsqueeze(0).float()

def count_metric(dir_res, dir_gt):
    files = os.listdir(dir_gt)
    l1, l2, dcds, emds, fs, fidel = [], [], [], [], [], []
    l1p, l2p, dcdsp, emdsp, fsp, fidelp = [], [], [], [], [], []
    print(len(files))
    for f in files:
        print(f)
        res_path = os.path.join(dir_res, f)
        gt_path = os.path.join(dir_gt, f)
        res_c, gt_c = read_res_and_gt(res_path, gt_path)
        l1.append(l1_cd(np2tensor(res_c), np2tensor(gt_c)).item()*1e3)
        l2.append(l2_cd(np2tensor(res_c), np2tensor(gt_c)).item()*1e3)
        dcds.append(calc_dcd(np2tensor(res_c), np2tensor(gt_c)).item())
        emds.append(emd(np2tensor(res_c), np2tensor(gt_c)).item()*1e3 / res_c.shape[0])
        fs.append(f_score(res_c, gt_c))
        fidel.append(fidelity(res_c, gt_c).item()*1e3)

        l1p.append(l1_cd(np2tensor(del_zdm(res_c)), np2tensor(del_zdm(gt_c))).item()*1e3)
        l2p.append(l2_cd(np2tensor(del_zdm(res_c)), np2tensor(del_zdm(gt_c))).item()*1e3)
        dcdsp.append(calc_dcd(np2tensor(del_zdm(res_c)), np2tensor(del_zdm(gt_c))).item())
        emdsp.append(emd(np2tensor(del_zdm(res_c)), np2tensor(del_zdm(gt_c))).item()*1e3 / res_c.shape[0])
        fsp.append(f_score(del_zdm(res_c), del_zdm(gt_c)))
        fidelp.append(fidelity(del_zdm(res_c), del_zdm(gt_c)).item()*1e3)

    print('l1_cd:{:.3f}±{:.3f}, l2_cd:{:.3f}±{:.3f}, dcd:{:.3f}±{:.3f}, emd:{:.3f}±{:.3f}, f_score:{:.4f}±{:.4f}, fidelity:{:.3f}±{:.3f}'.format(
        np.mean(l1), np.std(l1), np.mean(l2), np.std(l2), np.mean(dcds), np.std(dcds),
        np.mean(emds), np.std(emds), np.mean(fs), np.std(fs), np.mean(fidel), np.std(fidel)
    ))
    print('l1_cd_zg:{:.3f}±{:.3f}, l2_cd_zg:{:.3f}±{:.3f}, dcd_zg:{:.3f}±{:.3f}, emd_zg:{:.3f}±{:.3f}, f_score_zg:{:.4f}±{:.4f}, fidelity_zg:{:.3f}±{:.3f}'.format(
        np.mean(l1p), np.std(l1p), np.mean(l2p), np.std(l2p), np.mean(dcdsp), np.std(dcdsp),
        np.mean(emdsp), np.std(emdsp), np.mean(fsp), np.std(fsp), np.mean(fidelp), np.std(fidelp)
    ))

# dir_gt = 'test_res/res/coronary/output_gt'
# dir_res = 'test_res/res/coronary/output_res'

count_metric(dir_res, dir_gt)

