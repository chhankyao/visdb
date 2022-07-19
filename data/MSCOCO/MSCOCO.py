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
from utils.preprocessing import load_img, process_bbox, augmentation, match_dp_bbox, check_occ, check_trunc
from utils.transforms import world2cam, cam2pixel, pixel2cam, transform_joint_to_other_db, img2bb, rigid_transform_3D
from utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
from gmm_prior import gmm
from dense_pose import dp
from config import cfg
import time


class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = 'train' if data_split == 'train' else 'val'
        self.img_dir = osp.join(cfg.data_dir, 'MSCOCO', 'images')
        self.ann_path = osp.join(cfg.data_dir, 'MSCOCO', 'annotations', 'person_keypoints_%s2017.json' % self.data_split)
        self.smpl_param_path = osp.join(cfg.data_dir, 'MSCOCO', 'annotations', 'coco_smplifyx_train.json')
        self.rootnet_output_path = osp.join(cfg.data_dir, 'MSCOCO', 'annotations', 'bbox_root_coco_output.json')
        self.joint_regressor_path = osp.join(cfg.data_dir, 'MSCOCO', 'J_regressor_coco_hip_smpl.npy')
        self.fitting_thr = 3.0

        # mscoco skeleton
        self.coco_joint_num = 18 # original: 17, manually added pelvis
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
                                 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
                                 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
                                 'L_Ankle', 'R_Ankle', 'Pelvis')
        self.coco_skeleton = ((1, 2), (0, 1), (0, 2), (2, 4), (1, 3), 
                              (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), 
                              (14, 16), (11, 13), (13, 15), (5, 6), (11, 12))
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_joint_regressor = np.load(self.joint_regressor_path)
        self.smpl_to_coco_joints = (17, 11, 12, 17, 13, 14, 17, 15, 16,  5,
                                    15, 16,  0,  5,  6,  0,  5,  6,  7,  8,
                                     9, 10,  9, 10,  0,  1,  2,  3,  4)
        if cfg.add_head_top:
            self.smpl_to_coco_joints += (0,)

        # smpl skeleton
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
        print("Number of MSCOCO examples = %d" % len(self.datalist))

    def add_pelvis(self, joint_coord, joint_valid=True):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        if joint_valid:
            pelvis[2] = min(joint_coord[lhip_idx, 2], joint_coord[rhip_idx, 2]) # joint_valid
        pelvis = pelvis.reshape(1,3)
        joint_coord = np.concatenate((joint_coord, pelvis))
        return joint_coord

    def load_data(self):
        db = COCO(self.ann_path)
        with open(self.smpl_param_path) as f:
            smplify_results = json.load(f)

        datalist = []
        if self.data_split == 'train':
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_dir, imgname)
                width, height = img['width'], img['height']
                
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue
                
                # bbox
                bbox = process_bbox(ann['bbox'], width, height, self.data_split, 'MSCOCO')
                if bbox is None: 
                    continue
                
                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                joint_img = self.add_pelvis(joint_img)
                
                if str(aid) in smplify_results:
                    smplify_result = smplify_results[str(aid)]
                else:
                    smplify_result = None

                datalist.append({
                    'img_id': str(ann['image_id']),
                    'ann_id': str(aid),
                    'img_path': img_path,
                    'img_shape': (height, width),
                    'bbox': bbox,
                    'bbox_orig': ann['bbox'],
                    'joint_img': joint_img,
                    'smplify_result': smplify_result
                })

        else:
            with open(self.rootnet_output_path) as f:
                rootnet_output = json.load(f)
            print('Load RootNet output from  ' + self.rootnet_output_path)
            for i in range(len(rootnet_output)):
                image_id = rootnet_output[i]['image_id']
                if image_id not in db.imgs:
                    continue
                img = db.loadImgs(image_id)[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_dir, imgname)
                height, width = img['height'], img['width']

                fx, fy, cx, cy = 1500, 1500, img['width']/2, img['height']/2
                focal = np.array([fx, fy], dtype=np.float32); princpt = np.array([cx, cy], dtype=np.float32);
                root_joint_depth = np.array(rootnet_output[i]['root_cam'][2])
                bbox = np.array(rootnet_output[i]['bbox']).reshape(4)
                cam_param = {'focal': focal, 'princpt': princpt}

                datalist.append({
                    'img_path': img_path,
                    'img_shape': (height, width),
                    'bbox': bbox,
                    'bbox_orig': bbox,
                    'root_joint_depth': root_joint_depth,
                    'cam_param': cam_param
                })
        
        if self.data_split == 'train':
            return datalist[::cfg.skip_coco]
        else:
            return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(1,-1); # smpl parameters (pose: 72 dimension)
        smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(1,-1) # translation vector
        
        # flip smpl pose parameter (axis-angle)
        if do_flip:
            smpl_pose = smpl_pose.view(-1, 3)
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
        is_valid_fit = log_likelihood < 300 and np.all(np.abs(smpl_shape.numpy()) < 10)
            
        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy(), smpl_trans[0].numpy(), is_valid_fit
    
    def get_fitting_error(self, orig_joint_img, smpl_joint_img, orig_joint_valid):
        error_per_joint = np.sqrt(np.sum((orig_joint_img[:,:2] - smpl_joint_img[:,:2])**2, 1, keepdims=True)) * orig_joint_valid
        error = np.sum(error_per_joint) / (np.sum(orig_joint_valid) + 1e-6)
        return error_per_joint, error

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, bbox_orig = data['img_path'], data['img_shape'], data['bbox'], data['bbox_orig']
        
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
            if do_flip:
                orig_joint_img[:,0] = img_shape[1] - 1 - orig_joint_img[:,0]
                for pair in self.coco_flip_pairs:
                    orig_joint_img[pair[0],:], orig_joint_img[pair[1],:] = orig_joint_img[pair[1],:].copy(), orig_joint_img[pair[0],:].copy()

            # affine transform x,y coordinates, root-relative depth
            orig_joint_img = img2bb(orig_joint_img, img2bb_trans, 0)
            
            # check truncation
            orig_joint_trunc = check_trunc(orig_joint_img, cfg.output_hm_shape[1:], is_3D=False, combine=True)
            orig_joint_valid = (orig_joint_img[:,2] > 0).reshape(-1,1).astype(np.float32) * orig_joint_trunc
            orig_joint_occ = (orig_joint_img[:,2] == 2).reshape(-1,1).astype(np.float32) * orig_joint_trunc
            is_full_body = np.all(orig_joint_trunc)
            orig_joint_img[:,0] = np.clip(orig_joint_img[:,0], 0, cfg.output_hm_shape[2])
            orig_joint_img[:,1] = np.clip(orig_joint_img[:,1], 0, cfg.output_hm_shape[1])
            orig_joint_img[:,2] = 0

            # transform coco joints to target db joints
            orig_joint_img = transform_joint_to_other_db(orig_joint_img, self.coco_joints_name, self.joints_name)
            orig_joint_valid = transform_joint_to_other_db(orig_joint_valid, self.coco_joints_name, self.joints_name)
            orig_joint_cam = orig_joint_img.copy() # np.zeros((self.joint_num,3), dtype=np.float32)
            orig_joint_occ = orig_joint_occ[self.smpl_to_coco_joints, :]
            
            # get smpl fitting results
            smplify_result = data['smplify_result'] 
            if smplify_result is not None:
                smpl_param, cam_param = smplify_result['smpl_param'], smplify_result['cam_param']
                focal, princpt = cam_param['focal'], cam_param['princpt']
                
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
                smpl_joint_occ = smpl_trunc[self.vertex_num:] * smpl_occ[self.vertex_num:] * orig_joint_occ
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
                rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
                                        [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                        [0, 0, 1]], dtype=np.float32)
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
                error_per_joint, error = self.get_fitting_error(orig_joint_img, smpl_joint_img, orig_joint_valid)
                if error > self.fitting_thr:
                    is_valid_fit = False
                
            else:
                smpl_pose = np.zeros((72), dtype=np.float32)
                smpl_shape = np.zeros((10), dtype=np.float32)
                smpl_mesh_cam = np.zeros((self.vertex_num, 3), dtype=np.float32)
                smpl_joint_cam = np.zeros((self.joint_num, 3), dtype=np.float32)
                smpl_joint_img = np.zeros((self.joint_num, 3), dtype=np.float32)
                smpl_mesh_img = np.zeros((self.vertex_num, 3), dtype=np.float32)
                smpl_joint_vis = np.zeros((self.joint_num, 4), dtype=np.float32)
                smpl_mesh_vis = np.zeros((self.vertex_num, 4), dtype=np.float32)
                focal = [2000, 2000]
                princpt = [img_shape[1]//2, img_shape[0]//2]
                root_joint_depth = 1
                is_valid_fit = False
                error = 0
            
            inputs = {'img': img}
            targets = {
                'orig_joint_cam': orig_joint_cam, # (29, 3)
                'orig_joint_img': orig_joint_img, # (29, 3)
                'fit_joint_cam': smpl_joint_cam,  # (29, 3)
                'fit_joint_img': smpl_joint_img,  # (29, 3)
                'fit_mesh_img': smpl_mesh_img,    # (6890, 3)
                'pose_param': smpl_pose,          # (72,)
                'shape_param': smpl_shape,        # (10,)
                'dps': dps                        # (6890, 3)
            }
            meta_info = {
                'orig_joint_valid': orig_joint_valid, # (29, 1)
                'fit_joint_vis': smpl_joint_vis,      # (29, 4)
                'fit_mesh_vis': smpl_mesh_vis,        # (6890, 4)
                'is_3D': float(False),
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
            
        else:
            # smpl coordinates
            smpl_mesh_cam, smpl_joint_cam, trans = self.get_smpl_coord2(smpl_param)
            root_joint_depth = data['root_joint_depth'] if data['root_joint_depth'] is not None else 1
            
            inputs = {'img': img}
            targets = {
                'fit_mesh_coord_cam': smpl_mesh_cam,
                'dps': dps
            }
            meta_info = {
                'bb2img_trans': bb2img_trans,
                'img2bb_trans': img2bb_trans,
                'focal': np.array(focal, dtype=np.float32),
                'princpt': np.array(princpt, dtype=np.float32),
                'root_joint_depth': root_joint_depth,
                'bbox': np.array(bbox_orig, dtype=np.float32)
            }
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            out = outs[n]
            
            # 3D joint annotations
            orig_joint_cam = annot['joint_cam']
            orig_joint_cam = orig_joint_cam - orig_joint_cam[self.root_joint_idx,None,:]
            
            # h36m joint from gt mesh
            mesh_gt_cam = out['mesh_coord_cam_target']
            mesh_gt_img = cam2pixel(mesh_gt_cam, focal, princpt)
            joint_gt_cam = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            joint_gt_img = cam2pixel(joint_gt_cam, focal, princpt)
            root_joint_depth_gt = joint_gt_cam[self.h36m_root_joint_idx,2]
            joint_gt_cam_root = joint_gt_cam[self.h36m_root_joint_idx,None]
            joint_gt_cam = joint_gt_cam - joint_gt_cam_root # root-relative
            joint_gt_cam = joint_gt_cam[self.h36m_eval_joint,:]
            
            # visibility
            x1, y1 = annot['bbox'][0], annot['bbox'][1]
            x2, y2 = annot['bbox'][0]+annot['bbox'][2], annot['bbox'][1]+annot['bbox'][3]
            joint_trunc_gt = ((joint_gt_img[:,0] >= x1) * (joint_gt_img[:,0] < x2) * \
                              (joint_gt_img[:,1] >= y1) * (joint_gt_img[:,1] < y2))[np.array(self.h36m_eval_joint)]
            mesh_trunc_gt = (mesh_gt_img[:,0] >= x1) * (mesh_gt_img[:,0] < x2) * \
                            (mesh_gt_img[:,1] >= y1) * (mesh_gt_img[:,1] < y2)
            mesh_occ_gt = mesh_trunc_gt * out['dps'][:,2]
            if cfg.predict_vis:
                smpl_joint_gt_cam = np.dot(self.joint_regressor, mesh_gt_cam)
                smpl_joint_gt_img = cam2pixel(smpl_joint_gt_cam, focal, princpt)
                smpl_joint_trunc_gt = (smpl_joint_gt_img[:,0] >= 0) * (smpl_joint_gt_img[:,0] < annot['img_shape'][1]) * \
                                      (smpl_joint_gt_img[:,1] >= 0) * (smpl_joint_gt_img[:,1] < annot['img_shape'][0])
                joint_trunc_pred = (out['joint_vis'][:,0]>=0.5) * (out['joint_vis'][:,1]>=0.5)
                joint_occ_pred = (out['joint_vis'][:,2]>=0.5) * joint_trunc_pred
                mesh_trunc_pred = (out['vert_vis'][:,0]>0.5) * (out['vert_vis'][:,1]>0.5)
                mesh_occ_pred = (out['vert_vis'][:,2]>0.5) * mesh_trunc_pred         
                
            # mesh from lixel
            mesh_lixel_bb = out['mesh_coord_img']
            joint_lixel_bb = out['joint_coord_img']
            mesh_lixel_bb[:,0] = mesh_lixel_bb[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            mesh_lixel_bb[:,1] = mesh_lixel_bb[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            joint_lixel_bb[:,0] = joint_lixel_bb[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            joint_lixel_bb[:,1] = joint_lixel_bb[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            mesh_lixel_img = mesh_lixel_bb.copy()
            mesh_lixel_img_xy1 = np.concatenate((mesh_lixel_img[:,:2], np.ones_like(mesh_lixel_img[:,:1])),1)
            mesh_lixel_img[:,:2] = np.dot(out['bb2img_trans'], mesh_lixel_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            if cfg.use_gt_info:
                root_joint_depth = root_joint_depth_gt
            else:
                root_joint_depth = annot['root_joint_depth']
            mesh_lixel_img[:,2] = (mesh_lixel_img[:,2] / cfg.output_hm_shape[0]*2.-1) * (cfg.bbox_3d_size/2)
            mesh_lixel_img[:,2] = mesh_lixel_img[:,2] + root_joint_depth
            mesh_lixel_cam = pixel2cam(mesh_lixel_img, focal, princpt)

            if cfg.stage == 'param':
                mesh_param_cam = out['mesh_coord_cam']
                mesh_param_cam_reg = out['mesh_coord_cam_reg']
                root_xy = np.dot(self.joint_regressor, mesh_lixel_img)[self.root_joint_idx,:2]
                root_img = np.array([root_xy[0], root_xy[1], root_joint_depth])
                root_cam = pixel2cam(root_img[None,:], focal, princpt)
                mesh_param_cam_reg = mesh_param_cam_reg + root_cam.reshape(1,3)
        return None

    def print_eval_result(self, eval_result):
        pass 
