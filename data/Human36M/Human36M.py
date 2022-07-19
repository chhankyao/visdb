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


class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'Human36M', 'images')
        self.ann_dir = osp.join(cfg.data_dir, 'Human36M', 'annotations')
        self.rootnet_output_path = osp.join(self.ann_dir, 'bbox_root_human36m_output.json')
        self.joint_regressor_path = osp.join(cfg.data_dir, 'Human36M', 'J_regressor_h36m_correct.npy')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 
                            'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 
                            'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.fitting_thr = 0.04

        # H36M joint set
        self.h36m_joint_num = 17
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',
                                 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
                                 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder',
                                 'R_Elbow', 'R_Wrist')
        self.h36m_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.h36m_skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), 
                              (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), 
                              (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(self.joint_regressor_path)
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
        print('Number of Human3.6M examples = %d' % len(self.datalist))
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.ann_dir, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.ann_dir, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.ann_dir, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.ann_dir, 'Human36M_subject' + str(subject) + '_smpl_param.json'),'r') as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()

        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']), 
                                                               'root': np.array(annot[i]['root_cam'])}
        else:
            print("Get bounding box and root from groundtruth")

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img_id = ann['image_id']
            img = db.loadImgs(img_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; 
            subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            except KeyError:
                smpl_param = None
                
            if self.data_split == 'test' and smpl_param is None:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32)
            f, c = np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)
        
            if self.data_split == 'test' and not cfg.use_gt_info:
                bbox = bbox_root_result[str(img_id)]['bbox'] # bbox should be aspect ratio preserved-extended.
                root_joint_depth = bbox_root_result[str(img_id)]['root'][2]
            else:
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'], self.data_split, 'Human36M')
                if bbox is None:
                    continue
                root_joint_depth = joint_cam[self.h36m_root_joint_idx][2]
    
            datalist.append({
                'img_id': str(img_id),
                'ann_id': str(aid),
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'bbox_orig': ann['bbox'],
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'smpl_param': smpl_param,
                'root_joint_depth': root_joint_depth/1000
            })
        
        if self.data_split == 'train':
            return datalist[::cfg.skip_h36m]
        else:
            return datalist

    def get_smpl_coord(self, smpl_param, cam_param, do_flip, img_shape):
        pose, shape, trans = smpl_param['pose'], smpl_param['shape'], smpl_param['trans']
        smpl_pose = torch.FloatTensor(pose).view(-1,3) # smpl parameters (pose: 72 dimension)
        smpl_shape = torch.FloatTensor(shape).view(1,-1) # smpl parameters (shape: 10 dimension)
        R = np.array(cam_param['R'], dtype=np.float32).reshape(3,3) # camera rotation
        t = np.array(cam_param['t'], dtype=np.float32).reshape(3) # camera translation
        
        # merge root pose and camera rotation 
        root_pose = smpl_pose[self.root_joint_idx, :].numpy()
        root_pose, _ = cv2.Rodrigues(root_pose)
        root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
        smpl_pose[self.root_joint_idx] = torch.from_numpy(root_pose).view(3)

        # flip smpl pose parameter (axis-angle)
        if do_flip:
            for pair in self.flip_pairs:
                # face keypoints are already included in self.flip_pairs. However, they are not included in smpl_pose.
                if pair[0] < len(smpl_pose) and pair[1] < len(smpl_pose):
                    smpl_pose[pair[0],:], smpl_pose[pair[1],:] = smpl_pose[pair[1],:].clone(), smpl_pose[pair[0],:].clone()
            smpl_pose[:,1:3] *= -1; # multiply -1 to y and z axis of axis-angle
        smpl_pose = smpl_pose.view(1,-1)
       
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer['neutral'](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3); 
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
 
        # compenstate rotation (translation from origin to root joint was not cancled)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(3)
        smpl_trans = np.dot(R, smpl_trans[:,None]).reshape(1,3) + t.reshape(1,3)/1000
        root_joint_coord = smpl_joint_coord[self.root_joint_idx].reshape(1,3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1,0)).transpose(1,0)
        smpl_mesh_coord = smpl_mesh_coord + smpl_trans
        smpl_joint_coord = smpl_joint_coord + smpl_trans

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
        return smpl_mesh_coord, smpl_joint_coord, smpl_pose[0].numpy(), smpl_shape[0].numpy(), smpl_trans[0], is_valid_fit
    
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
            orig_joint_cam = orig_joint_cam - orig_joint_cam[self.h36m_root_joint_idx,None,:]
            orig_joint_img[:,2] /= 1000
            
            if do_flip:
                orig_joint_cam[:,0] = -orig_joint_cam[:,0]
                orig_joint_img[:,0] = img_shape[1] - 1 - orig_joint_img[:,0]
                for pair in self.h36m_flip_pairs:
                    orig_joint_img[pair[0],:], orig_joint_img[pair[1],:] = orig_joint_img[pair[1],:].copy(), orig_joint_img[pair[0],:].copy()
                    orig_joint_cam[pair[0],:], orig_joint_cam[pair[1],:] = orig_joint_cam[pair[1],:].copy(), orig_joint_cam[pair[0],:].copy()
            
            # affine transform x,y coordinates, root-relative depth
            orig_joint_img = img2bb(orig_joint_img, img2bb_trans, orig_joint_img[self.h36m_root_joint_idx][2])
            
            # check truncation
            orig_joint_trunc = check_trunc(orig_joint_img, cfg.output_hm_shape, combine=True)
            orig_joint_valid = np.ones((self.h36m_joint_num,1)) * orig_joint_trunc
            is_full_body = np.all(orig_joint_trunc)
            orig_joint_img[:,0] = np.clip(orig_joint_img[:,0], 0, cfg.output_hm_shape[2])
            orig_joint_img[:,1] = np.clip(orig_joint_img[:,1], 0, cfg.output_hm_shape[1])
            orig_joint_img[:,2] = np.clip(orig_joint_img[:,2], 0, cfg.output_hm_shape[0])

            # transform h36m joints to target db joints
            orig_joint_img = transform_joint_to_other_db(orig_joint_img, self.h36m_joints_name, self.joints_name)
            orig_joint_cam = transform_joint_to_other_db(orig_joint_cam, self.h36m_joints_name, self.joints_name)
            orig_joint_valid = transform_joint_to_other_db(orig_joint_valid, self.h36m_joints_name, self.joints_name)
            
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
                'orig_joint_valid': orig_joint_valid,
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
            
        else:
            # smpl coordinates
            smpl_mesh_cam, _, _, _, trans, _ = self.get_smpl_coord(smpl_param, cam_param, do_flip, img_shape)
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
                'bbox': np.array(bbox_orig, dtype=np.float32),
                'occ_box': np.array(occ_bbox, dtype=np.float32)
            }
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe_lixel': [], 'pa_mpjpe_lixel': [], 'mpjpe_lixel_partial': [], 'pa_mpjpe_lixel_partial': [],
                       'mpvpe_lixel': [], 'pa_mpvpe_lixel': [], 'mpvpe_lixel_partial': [], 'pa_mpvpe_lixel_partial': [],
                       'mpjpe_param': [], 'pa_mpjpe_param': [], 'mpjpe_param_partial': [], 'pa_mpjpe_param_partial': [],
                       'mpvpe_param': [], 'pa_mpvpe_param': [], 'mpvpe_param_partial': [], 'pa_mpvpe_param_partial': [],
                       'joint_trunc': [], 'joint_occ': [], 'mesh_trunc': [], 'mesh_occ': []}
        
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            cam_param = annot['cam_param']
            focal, princpt = cam_param['focal'], cam_param['princpt']
            out = outs[n]
            
            # h36m joint from gt mesh
            mesh_gt_cam = out['mesh_coord_cam_target']
            mesh_gt_img = cam2pixel(mesh_gt_cam, focal, princpt)
            joint_gt_cam = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            joint_gt_img = cam2pixel(joint_gt_cam, focal, princpt)
            root_joint_depth_gt = joint_gt_cam[self.h36m_root_joint_idx,2]
            joint_gt_cam_root = joint_gt_cam[self.h36m_root_joint_idx,None]
            joint_gt_cam = joint_gt_cam - joint_gt_cam_root
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
                eval_result['joint_trunc'].append((joint_trunc_pred == smpl_joint_trunc_gt).mean())
                eval_result['joint_occ'].append((joint_occ_pred == smpl_joint_trunc_gt).mean())
                eval_result['mesh_trunc'].append((mesh_trunc_pred == mesh_trunc_gt).mean())
                eval_result['mesh_occ'].append((mesh_trunc_pred == mesh_occ_gt).mean())
            else:
                eval_result['joint_trunc'].append(0)
                eval_result['joint_occ'].append(0)
                eval_result['mesh_trunc'].append(0)
                eval_result['mesh_occ'].append(0)           
                
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
            mesh_lixel_img[:,2] = (mesh_lixel_img[:,2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)
            mesh_lixel_img[:,2] = mesh_lixel_img[:,2] + root_joint_depth
            mesh_lixel_cam = pixel2cam(mesh_lixel_img, focal, princpt)
            
            # h36m joint from lixel mesh
            joint_lixel_cam = np.dot(self.h36m_joint_regressor, mesh_lixel_cam)
            joint_lixel_img = cam2pixel(joint_lixel_cam, focal, princpt)
            joint_lixel_cam_root = joint_lixel_cam[self.h36m_root_joint_idx,None]
            joint_lixel_cam = joint_lixel_cam - joint_lixel_cam_root
            joint_lixel_cam = joint_lixel_cam[self.h36m_eval_joint,:]
            
            # joint error
            joint_lixel_cam_aligned = rigid_align(joint_lixel_cam, joint_gt_cam)
            mpjpe_lixel = np.sqrt(np.sum((joint_lixel_cam - joint_gt_cam)**2,1)) * 1000
            pa_mpjpe_lixel = np.sqrt(np.sum((joint_lixel_cam_aligned - joint_gt_cam)**2,1)) * 1000
            eval_result['mpjpe_lixel'].append(mpjpe_lixel[joint_trunc_gt].mean())
            eval_result['pa_mpjpe_lixel'].append(pa_mpjpe_lixel[joint_trunc_gt].mean())
            if not np.all(joint_trunc_gt):
                eval_result['mpjpe_lixel_partial'].append(mpjpe_lixel[joint_trunc_gt].mean())
                eval_result['pa_mpjpe_lixel_partial'].append(pa_mpjpe_lixel[joint_trunc_gt].mean())
            
            # vertex error
            _mesh_gt_cam = mesh_gt_cam - joint_gt_cam_root
            _mesh_lixel_cam = mesh_lixel_cam - joint_lixel_cam_root
            mesh_lixel_cam_aligned = rigid_align(_mesh_lixel_cam, _mesh_gt_cam)
            mpvpe_lixel = np.sqrt(np.sum((_mesh_lixel_cam - _mesh_gt_cam)**2,1)) * 1000
            pa_mpvpe_lixel = np.sqrt(np.sum((mesh_lixel_cam_aligned - _mesh_gt_cam)**2,1)) * 1000
            eval_result['mpvpe_lixel'].append(mpvpe_lixel[mesh_trunc_gt].mean())
            eval_result['pa_mpvpe_lixel'].append(pa_mpvpe_lixel[mesh_trunc_gt].mean())
            if not np.all(mesh_trunc_gt):
                eval_result['mpvpe_lixel_partial'].append(mpvpe_lixel[mesh_trunc_gt].mean())
                eval_result['pa_mpvpe_lixel_partial'].append(pa_mpvpe_lixel[mesh_trunc_gt].mean())
            
            # h36m joint from parameter mesh
            if cfg.stage == 'param':
                mesh_param_cam = out['mesh_coord_cam']
                mesh_param_cam_reg = out['mesh_coord_cam_reg']
                root_xy = np.dot(self.joint_regressor, mesh_lixel_img)[self.root_joint_idx,:2]
                root_img = np.array([root_xy[0], root_xy[1], root_joint_depth])
                root_cam = pixel2cam(root_img[None,:], focal, princpt)
                mesh_param_cam_reg = mesh_param_cam_reg + root_cam.reshape(1,3)
                
                joint_param_cam = np.dot(self.h36m_joint_regressor, mesh_param_cam)
                joint_param_cam_root = joint_param_cam[self.h36m_root_joint_idx,None]
                joint_param_cam = joint_param_cam - joint_param_cam_root
                joint_param_cam = joint_param_cam[self.h36m_eval_joint,:]
                joint_param_cam_aligned = rigid_align(joint_param_cam, joint_gt_cam)
                mpjpe_param = np.sqrt(np.sum((joint_param_cam - joint_gt_cam)**2,1)).mean() * 1000
                pa_mpjpe_param = np.sqrt(np.sum((joint_param_cam_aligned - joint_gt_cam)**2,1)).mean() * 1000
                eval_result['mpjpe_param'].append(mpjpe_param)
                eval_result['pa_mpjpe_param'].append(pa_mpjpe_param)
                if not np.all(joint_trunc_gt):
                    eval_result['mpjpe_param_partial'].append(mpjpe_param)
                    eval_result['pa_mpjpe_param_partial'].append(pa_mpjpe_param)
                
                _mesh_param_cam = mesh_param_cam - joint_param_cam_root
                mesh_param_cam_aligned = rigid_align(_mesh_param_cam, _mesh_gt_cam)
                mpvpe_param = np.sqrt(np.sum((_mesh_param_cam - _mesh_gt_cam)**2,1)).mean() * 1000
                pa_mpvpe_param = np.sqrt(np.sum((mesh_param_cam_aligned - _mesh_gt_cam)**2,1)).mean() * 1000
                eval_result['mpvpe_param'].append(mpvpe_param)
                eval_result['pa_mpvpe_param'].append(pa_mpvpe_param)
                if not np.all(mesh_trunc_gt):
                    eval_result['mpvpe_param_partial'].append(mpvpe_param)
                    eval_result['pa_mpvpe_param_partial'].append(pa_mpvpe_param)
            else:
                _mesh_param_cam = _mesh_lixel_cam
                joint_param_cam = joint_lixel_cam
                eval_result['mpjpe_param'].append(0)
                eval_result['pa_mpjpe_param'].append(0)
                eval_result['mpjpe_param_partial'].append(0)
                eval_result['pa_mpjpe_param_partial'].append(0)
                eval_result['mpvpe_param'].append(0)
                eval_result['pa_mpvpe_param'].append(0)
                eval_result['mpvpe_param_partial'].append(0)
                eval_result['pa_mpvpe_param_partial'].append(0)                
        return eval_result

    def print_eval_result(self, eval_result):
        with open('../output/%s/test_%s_%d.txt' % (cfg.exp_name, cfg.stage, len(eval_result['mpjpe_lixel'])), 'w') as f:
            print('number of total examples: %d' % len(eval_result['mpjpe_lixel']))
            print('number of partial-joint examples: %d' % len(eval_result['mpjpe_lixel_partial']))
            print('number of partial-mesh examples: %d' % len(eval_result['mpvpe_lixel_partial']))
            f.write('number of total examples: %d\n' % len(eval_result['mpjpe_lixel']))
            f.write('number of partial-joint examples: %d\n' % len(eval_result['mpjpe_lixel_partial']))
            f.write('number of partial-mesh examples: %d\n' % len(eval_result['mpvpe_lixel_partial']))
            for k, v in eval_result.items():
                print('%s = %.2f' % (k, np.mean(v)))
                f.write('%s = %.2f\n' % (k, np.mean(v)))
                