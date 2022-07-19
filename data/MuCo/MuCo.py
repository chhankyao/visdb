import os
import os.path as osp
import numpy as np
import pickle
import json
import cv2
import random
import math
import copy
import torch
import transforms3d
import scipy.io as sio
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, match_dp_bbox, check_occ, check_trunc
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, img2bb, rigid_transform_3D
from utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
from gmm_prior import gmm
from dense_pose import dp
from config import cfg


class MuCo(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'MuCo', 'images')
        self.ann_path = osp.join(cfg.data_dir, 'MuCo', 'annotations', 'MuCo-3DHP.json')
        self.smpl_param_path = osp.join(cfg.data_dir, 'MuCo', 'annotations', 'smpl_param.json')
        self.joint_regressor_path = osp.join(cfg.data_dir, 'Human36M', 'J_regressor_h36m_correct.npy')
        self.fitting_thr = 0.06

        # MuCo joint set
        self.muco_joint_num = 21
        self.muco_joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
                                 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
                                 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis',
                                 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe',
                                 'L_Toe')
        self.muco_flip_pairs = ((2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (17, 18), (19, 20))
        self.muco_skeleton = ((0, 16), (16, 1), (1, 15), (15, 14), (14, 8), 
                              (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), 
                              (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), 
                              (4, 17), (1, 5), (5, 6), (6, 7), (7, 18))
        self.muco_root_joint_idx = self.muco_joints_name.index('Pelvis')
        
        # H36M joint set
        self.h36m_joint_regressor = np.load(self.joint_regressor_path)
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',
                                 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
                                 'Head_Top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder',
                                 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.smpl_to_muco_joints = (14, 11, 8, 14, 12, 9, 15, 13, 10, 15,
                                    20, 19, 1, 5, 2, 0, 5, 2, 6, 3,
                                    7, 4, 18, 17, 16, 16, 16, 16, 16)
        self.smpl_to_h36m_joints = (0, 4, 1, 0, 5, 2, 7, 6, 3, 7, 
                                    6, 3, 8, 11, 14, 10, 11, 14, 12, 15, 
                                    13, 16, 13, 16, 9, 9, 9, 9, 9)

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.flip_pairs = self.smpl.flip_pairs
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.datalist = self.load_data()
        print('Number of MuCo examples = %d' % len(self.datalist))
        
    def load_data(self):
        if self.data_split == 'train':
            db = COCO(self.ann_path)
            with open(self.smpl_param_path) as f:
                smpl_params = json.load(f)
        else:
            print('Unknown data subset')
            assert 0

        datalist = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img['id']
            img_width, img_height = img['width'], img['height']
            imgname = img['file_name']
            img_path = osp.join(self.img_dir, imgname)
            focal = img['f']
            princpt = img['c']
            cam_param = {'focal': focal, 'princpt': princpt}

            # crop the closest person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            root_depths = [ann['keypoints_cam'][self.muco_root_joint_idx][2] for ann in anns]
            closest_pid = root_depths.index(min(root_depths))
            pid_list = [closest_pid]
            for pid in pid_list:
                joint_cam = np.array(anns[pid]['keypoints_cam'])
                joint_img = np.array(anns[pid]['keypoints_img'])
                joint_img = np.concatenate([joint_img, joint_cam[:,2:]],1)

                bbox = process_bbox(anns[pid]['bbox'], img_width, img_height, self.data_split, 'MuCo')
                if bbox is None: continue
                
                # check smpl parameter exist
                try:
                    smpl_param = smpl_params[str(ann_ids[pid])]
                except KeyError:
                    smpl_param = None

                datalist.append({
                    'img_id': str(img_id),
                    'ann_id': str(ann_ids[pid]),
                    'img_path': img_path,
                    'img_shape': (img_height, img_width),
                    'bbox': bbox,
                    'bbox_orig': anns[pid]['bbox'],
                    'joint_img': joint_img,
                    'joint_cam': joint_cam,
                    'cam_param': cam_param,
                    'smpl_param': smpl_param
                })
        
        return datalist[::cfg.skip_muco]

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1,-1) # smpl parameters (pose: 72 dimension)
        smpl_shape = torch.FloatTensor(shape).view(1,-1) # smpl parameters (shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1,3)
       
        # flip smpl pose parameter (axis-angle)
        if do_flip:
            smpl_pose = smpl_pose.view(-1,3)
            for pair in self.flip_pairs:
                # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):
                    smpl_pose[pair[0],:], smpl_pose[pair[1],:] = smpl_pose[pair[1],:].clone(), smpl_pose[pair[0],:].clone()
            smpl_pose[:,1:3] *= -1; # multiply -1 to y and z axis of axis-angle
            smpl_pose = smpl_pose.view(1,-1)
        
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape, smpl_trans)
       
        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3); 
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))

        # flip translation
        if do_flip: # avg of old and new root joint should be image center.
            focal, princpt = cam_param['focal'], cam_param['princpt']
            flip_trans_x = 2 * (((img_shape[1] - 1)/2. - princpt[0]) / focal[0] * smpl_joint_coord[self.root_joint_idx,2]) - \
                           2 * smpl_joint_coord[self.root_joint_idx][0]
            smpl_mesh_coord[:,0] += flip_trans_x
            smpl_joint_coord[:,0] += flip_trans_x
            smpl_trans[:,0] += flip_trans_x

        log_likelihood = gmm(smpl_pose)[0].numpy()
        is_valid_fit = log_likelihood < 800 and np.all(np.abs(smpl_shape.numpy()) < 10)
            
        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy(), smpl_trans[0].numpy(), is_valid_fit

    def get_fitting_error(self, orig_joint_cam, smpl_joint_cam, orig_joint_valid):
        _orig_joint_valid = orig_joint_valid.copy()
        _orig_joint_valid[self.joints_name.index('L_Hip'), :] = 0
        _orig_joint_valid[self.joints_name.index('R_Hip'), :] = 0
        error_per_joint = np.sqrt(np.sum((orig_joint_cam - smpl_joint_cam)**2, 1, keepdims=True)) * _orig_joint_valid
        error = np.sum(error_per_joint) / (np.sum(_orig_joint_valid) + 1e-6)
        return error_per_joint, error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, bbox_orig = data['img_path'], data['img_shape'], data['bbox'], data['bbox_orig']
        smpl_param, cam_param = data['smpl_param'], data['cam_param']
        focal, princpt = cam_param['focal'], cam_param['princpt']
        
        # load dense pose
        bbox_iou = 0
        bbox_dp = None
        is_valid_dp = False
        dp_iuv = np.zeros((img_shape[0], img_shape[1], 3))
        if cfg.use_dp or cfg.predict_vis:
            dp_path = img_path.replace('images', 'dp').replace('.jpg', '.pkl')
            if osp.isfile(dp_path):
                dp_data = pickle.load(open(dp_path, 'rb'))
                idx_matched, bbox_iou, bbox_dp = match_dp_bbox(bbox_orig, dp_data['pred_boxes_XYXY'], img_shape)
                if bbox_iou > 0.5:
                    dp_img = cv2.imread(dp_path.replace('.pkl', '_%d.png' % idx_matched))
                    dp_iuv[bbox_dp[1]:bbox_dp[1]+dp_img.shape[0], bbox_dp[0]:bbox_dp[0]+dp_img.shape[1], :] = dp_img
                    is_valid_dp = True
        
        # load image and affine transform
        img_raw = load_img(img_path)
        img, dp_iuv, img2bb_trans, bb2img_trans, rot, do_flip, occ_bbox = augmentation(img_raw, dp_iuv, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # dense points from DensePose UV
        if is_valid_dp:
            if do_flip:
                dp_iuv = dp.flip_iuv(dp_iuv)
            dps = dp.dense_point_corr(dp_iuv)
        else:
            dps = np.zeros((self.vertex_num, 3), dtype=np.float32)
        
        if self.data_split == 'train':
            # joint annotations
            orig_joint_img = data['joint_img']
            orig_joint_cam = data['joint_cam']/1000
            orig_joint_cam = orig_joint_cam - orig_joint_cam[self.muco_root_joint_idx,None,:]
            orig_joint_img[:,2] /= 1000

            if do_flip:
                orig_joint_cam[:,0] = -orig_joint_cam[:,0]
                orig_joint_img[:,0] = img_shape[1] - 1 - orig_joint_img[:,0]
                for pair in self.muco_flip_pairs:
                    orig_joint_img[pair[0],:], orig_joint_img[pair[1],:] = orig_joint_img[pair[1],:].copy(), orig_joint_img[pair[0],:].copy()
                    orig_joint_cam[pair[0],:], orig_joint_cam[pair[1],:] = orig_joint_cam[pair[1],:].copy(), orig_joint_cam[pair[0],:].copy()
                    
            # affine transform x,y coordinates, root-relative depth
            orig_joint_img = img2bb(orig_joint_img, img2bb_trans, orig_joint_img[self.muco_root_joint_idx][2])
            
            # check truncation
            orig_joint_trunc = check_trunc(orig_joint_img, cfg.output_hm_shape, combine=True)
            orig_joint_valid = np.ones((self.muco_joint_num,1)) * orig_joint_trunc
            is_full_body = np.all(orig_joint_trunc)
            orig_joint_img[:,0] = np.clip(orig_joint_img[:,0], 0, cfg.output_hm_shape[2])
            orig_joint_img[:,1] = np.clip(orig_joint_img[:,1], 0, cfg.output_hm_shape[1])
            orig_joint_img[:,2] = np.clip(orig_joint_img[:,2], 0, cfg.output_hm_shape[0])
            
            # transform muco joints to target db joints
            orig_joint_img = transform_joint_to_other_db(orig_joint_img, self.muco_joints_name, self.joints_name)
            orig_joint_cam = transform_joint_to_other_db(orig_joint_cam, self.muco_joints_name, self.joints_name)
            orig_joint_valid = transform_joint_to_other_db(orig_joint_valid, self.muco_joints_name, self.joints_name)
            
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            orig_joint_cam = np.dot(rot_aug_mat, orig_joint_cam.transpose(1,0)).transpose(1,0)
            
            # get smpl fitting results
            if smpl_param is not None:
                smpl_mesh_cam, smpl_joint_cam, smpl_pose, smpl_shape, trans, is_valid_fit = self.get_smpl_coord(
                    smpl_param, cam_param, do_flip, img_shape)               
                smpl_coord_cam = np.concatenate((smpl_mesh_cam, smpl_joint_cam))
                smpl_coord_img = cam2pixel(smpl_coord_cam, focal, princpt)
                
                smpl_occ = check_trunc(smpl_coord_img, img_shape, is_3D=False, combine=True)
                smpl_occ *= check_occ(smpl_coord_img, occ_bbox)

                # affine transform x,y coordinates, root-relative depth
                smpl_coord_img = img2bb(smpl_coord_img, img2bb_trans, smpl_joint_cam[self.root_joint_idx][2])
                if np.sum(orig_joint_valid) < 5:
                    is_valid_fit = False
                
                # check truncation
                smpl_trunc_x, smpl_trunc_y, smpl_trunc_z = check_trunc(smpl_coord_img, cfg.output_hm_shape)
                smpl_trunc = smpl_trunc_x * smpl_trunc_y * smpl_trunc_z
                smpl_coord_img[:,0] = np.clip(smpl_coord_img[:,0], 0, cfg.output_hm_shape[2])
                smpl_coord_img[:,1] = np.clip(smpl_coord_img[:,1], 0, cfg.output_hm_shape[1])
                smpl_coord_img[:,2] = np.clip(smpl_coord_img[:,2], 0, cfg.output_hm_shape[0])
                
                # split mesh and joint coordinates
                smpl_mesh_img = smpl_coord_img[:self.vertex_num]
                smpl_joint_img = smpl_coord_img[self.vertex_num:]
                
                # visibility
                smpl_mesh_occ = smpl_trunc[:self.vertex_num] * smpl_occ[:self.vertex_num]
                smpl_joint_occ = smpl_trunc[self.vertex_num:] * smpl_occ[self.vertex_num:]
                if is_valid_dp:
                    face_occ = np.maximum(np.maximum(dps[self.face[:,0],2], dps[self.face[:,1],2]), dps[self.face[:,2],2])
                    vert_occ = np.zeros(dps.shape[0])
                    vert_occ[self.face[:,0]] = face_occ
                    vert_occ[self.face[:,1]] = face_occ
                    vert_occ[self.face[:,2]] = face_occ
                    smpl_mesh_occ *= vert_occ[:,None]
                smpl_mesh_vis = np.concatenate((smpl_trunc_x[:self.vertex_num], smpl_trunc_y[:self.vertex_num], 
                                                smpl_mesh_occ, np.clip(smpl_trunc[:self.vertex_num], 0.1, 1.0)), 1)
                smpl_joint_vis = np.concatenate((smpl_trunc_x[self.vertex_num:], smpl_trunc_y[self.vertex_num:], 
                                                 smpl_joint_occ, np.clip(smpl_trunc[self.vertex_num:], 0.1, 1.0)), 1)
                
                # 3D data rotation augmentation
                smpl_pose = smpl_pose.reshape(-1,3)
                root_pose = smpl_pose[self.root_joint_idx,:]
                root_pose, _ = cv2.Rodrigues(root_pose)
                root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
                smpl_pose[self.root_joint_idx] = root_pose.reshape(3)
                smpl_pose = smpl_pose.reshape(-1)
                smpl_root = smpl_joint_cam[self.root_joint_idx,None]
                smpl_joint_cam = smpl_joint_cam - smpl_root
                smpl_mesh_cam = smpl_mesh_cam - smpl_root
                smpl_joint_cam = np.dot(rot_aug_mat, smpl_joint_cam.transpose(1,0)).transpose(1,0)
                smpl_mesh_cam = np.dot(rot_aug_mat, smpl_mesh_cam.transpose(1,0)).transpose(1,0)
                root_joint_depth = smpl_root[0,2]

                # if fitted mesh is too far from gt, discard it
                error_per_joint, error = self.get_fitting_error(orig_joint_cam, smpl_joint_cam, orig_joint_valid)
                if error > self.fitting_thr:
                    is_valid_fit = False
            
            else:
                smpl_pose = np.zeros((72), dtype=np.float32)
                smpl_shape = np.zeros((10), dtype=np.float32)
                smpl_joint_cam = np.zeros((self.joint_num,3), dtype=np.float32)
                smpl_joint_img = np.zeros((self.joint_num,3), dtype=np.float32)
                smpl_mesh_img = np.zeros((self.vertex_num,3), dtype=np.float32)
                smpl_joint_vis = np.zeros((self.joint_num,4), dtype=np.float32)
                smpl_mesh_vis = np.zeros((self.vertex_num,4), dtype=np.float32)
                focal = [2000, 2000]
                princpt = [img_shape[1]//2, img_shape[0]//2]
                root_joint_depth = 1
                is_valid_fit = False
                error = 0
        
            inputs = {'img': img}
            targets = {
                'orig_joint_cam': orig_joint_cam,
                'orig_joint_img': orig_joint_img,
                'fit_joint_cam': smpl_joint_cam,
                'fit_joint_img': smpl_joint_img,
                'fit_mesh_img': smpl_mesh_img,
                'pose_param': smpl_pose,
                'shape_param': smpl_shape,
                'dps': dps
            }
            meta_info = {
                'orig_joint_valid': orig_joint_valid * float(is_valid_fit),
                'fit_joint_vis': smpl_joint_vis,
                'fit_mesh_vis': smpl_mesh_vis,
                'is_3D': float(True),
                'use_dp': 1 - float(is_valid_fit),
                'is_valid_dp': float(is_valid_dp),
                'is_valid_fit': float(is_valid_fit),
                'is_full_body': float(is_full_body),
                'bb2img_trans': bb2img_trans,
                'img2bb_trans': img2bb_trans,
                'focal': np.array(focal, dtype=np.float32),
                'princpt': np.array(princpt, dtype=np.float32),
                'root_joint_depth': root_joint_depth,
            }
        return inputs, targets, meta_info
