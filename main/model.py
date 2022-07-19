import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import PoseNet, Pose2Feat, MeshNet, ParamRegressor
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext

import time
import os.path as osp
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from gmm_prior import MaxMixturePrior
from dense_pose import dp


class Model(nn.Module):
    def __init__(self, pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor):
        super(Model, self).__init__()
        self.pose_backbone = pose_backbone
        self.pose_net = pose_net
        self.pose2feat = pose2feat
        self.mesh_backbone = mesh_backbone
        self.mesh_net = mesh_net
        self.param_regressor = param_regressor

        self.human_model = SMPL()
        self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx
        self.mesh_face = self.human_model.face
        self.joint_regressor = self.human_model.joint_regressor
        self.gmm = MaxMixturePrior(cfg.gmm_path).cuda()

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()
        self.normal_loss = NormalVectorLoss(self.mesh_face)
        self.edge_loss = EdgeLengthLoss(self.mesh_face)
        if cfg.predict_vis:
            self.vis_loss = nn.BCELoss(reduction='none')
        self.num_parts = 14
        self.vert_part = torch.from_numpy(dp.parts14-1).cuda()

    def make_gaussian_heatmap(self, joint_coord_img):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        z = torch.arange(cfg.output_hm_shape[0])
        zz,yy,xx = torch.meshgrid(z,y,x)
        xx = xx[None,None,:,:,:].cuda().float();
        yy = yy[None,None,:,:,:].cuda().float();
        zz = zz[None,None,:,:,:].cuda().float();
        x = joint_coord_img[:,:,0,None,None,None];
        y = joint_coord_img[:,:,1,None,None,None];
        z = joint_coord_img[:,:,2,None,None,None];
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 - (((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
        return heatmap

    def bb2img(self, pixel_coord, bb2img_trans):
        pixel_coord[:,:,0] = pixel_coord[:,:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        pixel_coord[:,:,1] = pixel_coord[:,:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        pixel_coord_xy1 = torch.cat((pixel_coord[:,:,:2], torch.ones_like(pixel_coord[:,:,:1])), 2)
        pixel_coord[:,:,:2] = torch.bmm(bb2img_trans, pixel_coord_xy1.permute(0,2,1)).permute(0,2,1)[:,:,:2]
        pixel_coord[:,:,2] = (pixel_coord[:,:,2] / cfg.output_hm_shape[0]*2. - 1) * (cfg.bbox_3d_size/2)
        return pixel_coord

    def img2bb(self, pixel_coord, img2bb_trans, root_joint_depth):
        pixel_coord_xy1 = torch.cat((pixel_coord[:,:,:2], torch.ones_like(pixel_coord[:,:,:1])), 2)
        pixel_coord[:,:,:2] = torch.bmm(img2bb_trans, pixel_coord_xy1.permute(0,2,1)).permute(0,2,1)[:,:,:2]
        pixel_coord[:,:,0] = pixel_coord[:,:,0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        pixel_coord[:,:,1] = pixel_coord[:,:,1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        pixel_coord[:,:,2] = pixel_coord[:,:,2] - root_joint_depth[:,None]
        pixel_coord[:,:,2] = (pixel_coord[:,:,2] / (cfg.bbox_3d_size/2) + 1)/2. * cfg.output_hm_shape[0]
        return pixel_coord

    def cam2pixel(self, cam_coord, f, c):
        z = cam_coord[:,:,2]
        x = cam_coord[:,:,0] / z * f[:,0,None] + c[:,0,None]
        y = cam_coord[:,:,1] / z * f[:,1,None] + c[:,1,None]
        return torch.stack((x,y,z), 2)

    def pixel2cam(self, pixel_coord, f, c):
        z = pixel_coord[:,:,2]
        x = (pixel_coord[:,:,0] - c[:,0,None]) / f[:,0,None] * z
        y = (pixel_coord[:,:,1] - c[:,1,None]) / f[:,1,None] * z
        return torch.stack((x,y,z),2)

    def estimate_root(self, joint_img, joint_cam, focal, princpt):
        limb_joints1 = [16, 17, 1, 2]
        limb_joints2 = [18, 19, 4, 5]
        lengths_cam = torch.linalg.norm(joint_cam[:,limb_joints1,:2] - joint_cam[:,limb_joints2,:2], dim=2)
        lengths_img = torch.linalg.norm(joint_img[:,limb_joints1,:2] - joint_img[:,limb_joints2,:2], dim=2)
        root_z = ((lengths_cam / (lengths_img + 1e-9)).mean(1)[:,None] * focal).mean(1)
        root_x = (joint_img[:,:,0] - princpt[:,None,0]) * (joint_cam[:,:,2] + root_z[:,None])
        root_y = (joint_img[:,:,1] - princpt[:,None,1]) * (joint_cam[:,:,2] + root_z[:,None])
        root_x = (root_x / focal[:,None,0] - joint_cam[:,:,0]).mean(1)
        root_y = (root_y / focal[:,None,1] - joint_cam[:,:,1]).mean(1)
        return torch.stack((root_x, root_y, root_z), 1)
    
    def optimize_param(self, params, lr, n_iters, targets, losses, params_opt, prior=True):
        bs = targets['mesh_cam'].shape[0]
        trans, pose_global, pose, shape = params        
        optimizer = optim.Adam(params_opt, lr=lr)
        for i in range(n_iters):
            optimizer.zero_grad()
            pose_combined = torch.cat((pose_global, pose), 1)
            mesh_param_cam, _ = self.human_model_layer(pose_combined, shape, trans)
            joint_param_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(bs,1,1), mesh_param_cam)
            loss = ((mesh_param_cam - targets['mesh_cam']).abs() * targets['mesh_trunc'][:,:,None]).mean()
            loss += ((joint_param_cam - targets['joint_cam']).abs() * targets['joint_trunc'][:,:,None]).mean()
            if prior:
                loss += self.gmm(pose_combined).mean() * 1e-5           
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().data)
        return mesh_param_cam

    def forward(self, inputs, targets, info, mode):
        if cfg.stage == 'lixel':
            cm = nullcontext()
        else:
            cm = torch.no_grad()

        with cm:
            # posenet forward
            shared_img_feat, pose_img_feat = self.pose_backbone(inputs['img'])
            joint_lixel_img, joint_vis = self.pose_net(pose_img_feat)

            # make 3D heatmap from posenet output and convert to image feature
            with torch.no_grad():
                joint_heatmap = self.make_gaussian_heatmap(joint_lixel_img.detach())

            shared_img_feat = self.pose2feat(shared_img_feat, joint_heatmap, joint_vis)

            # meshnet forward
            _, mesh_img_feat = self.mesh_backbone(shared_img_feat, skip_early=True)
            mesh_lixel_img, mesh_vis = self.mesh_net(mesh_img_feat)

            # joint coordinate outputs from mesh coordinates
            bs = mesh_lixel_img.shape[0]
            joint_img_from_mesh = torch.bmm(
                torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(bs,1,1), mesh_lixel_img)

        if cfg.stage == 'param':
            # parameter regression
            if cfg.predict_vis:
                pose_param, shape_param = self.param_regressor(joint_img_from_mesh.detach(), mesh_lixel_img.detach(),
                                                               joint_vis.detach(), mesh_vis.detach())
            else:
                pose_param, shape_param = self.param_regressor(joint_img_from_mesh.detach(), mesh_lixel_img.detach())

            # get mesh and joint coordinates
            mesh_param_cam, _ = self.human_model_layer(pose_param, shape_param)
            joint_param_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(bs,1,1), mesh_param_cam)

            # root-relative 3D coordinates
            root_joint_cam = joint_param_cam[:, self.root_joint_idx, None, :]
            mesh_param_cam = mesh_param_cam - root_joint_cam
            joint_param_cam = joint_param_cam - root_joint_cam

        # training
        if mode == 'train':
            loss = {}

            if cfg.stage == 'lixel':
                is_3D = info['is_3D']
                use_dp = info['use_dp'][:,None,None]
                is_valid_dp = info['is_valid_dp'][:,None,None]
                is_valid_fit = info['is_valid_fit'][:,None,None]
                orig_joint_valid = info['orig_joint_valid']
                fit_joint_valid = info['fit_joint_vis'][:,:,3:] * is_valid_fit
                fit_mesh_valid = info['fit_mesh_vis'][:,:,3:] * is_valid_fit

                loss['joint'] = self.coord_loss(
                    joint_lixel_img, targets['orig_joint_img'], orig_joint_valid, is_3D) * cfg.w_joint
                loss['joint_fit'] = self.coord_loss(
                    joint_lixel_img, targets['fit_joint_img'], fit_joint_valid) * cfg.w_joint_fit
                loss['mesh_fit'] = self.coord_loss(
                    mesh_lixel_img, targets['fit_mesh_img'], fit_mesh_valid) * cfg.w_mesh_fit
                loss['mesh_edge'] = self.edge_loss(
                    mesh_lixel_img, targets['fit_mesh_img'], fit_mesh_valid) * cfg.w_mesh_edge
                loss['mesh_normal'] = self.normal_loss(
                    mesh_lixel_img, targets['fit_mesh_img'], fit_mesh_valid) * cfg.w_mesh_normal
                loss['mesh_joint'] = self.coord_loss(
                    joint_img_from_mesh, targets['orig_joint_img'], orig_joint_valid, is_3D) * cfg.w_mesh_joint
                loss['mesh_joint_fit'] = self.coord_loss(
                    joint_img_from_mesh, targets['fit_joint_img'], fit_joint_valid) * cfg.w_mesh_joint_fit
                if cfg.use_dp:
                    loss['dp'] = self.coord_loss(
                        mesh_lixel_img[:,:,:2], targets['dps'][:,:,:2], targets['dps'][:,:,2:] * use_dp) * cfg.w_dps
                if cfg.predict_vis:
                    loss['joint_vis'] = self.vis_loss(
                        joint_vis, info['fit_joint_vis'][:,:,:3]) * is_valid_fit * cfg.w_vis * 0.1
                    loss['mesh_vis_xy'] = self.vis_loss(
                        mesh_vis[:,:,:2], info['fit_mesh_vis'][:,:,:2]) * is_valid_fit * cfg.w_vis * 0.1
                    loss['mesh_vis_z'] = self.vis_loss(
                        mesh_vis[:,:,2], info['fit_mesh_vis'][:,:,2]) * is_valid_fit * is_valid_dp * cfg.w_vis

            else:
                is_3D = info['is_3D'][:,None,None]
                is_valid_fit = info['is_valid_fit'][:,None]
                orig_joint_valid = info['orig_joint_valid']

                if cfg.predict_vis:
                    loss['pose_param'] = self.param_loss(pose_param, targets['pose_param'], is_valid_fit) * 0.5
                    loss['shape_param'] = self.param_loss(shape_param, targets['shape_param'], is_valid_fit) * 0.5
                    loss['joint_orig_cam'] = self.coord_loss(joint_param_cam, targets['orig_joint_cam'], orig_joint_valid * is_3D)
                    loss['joint_fit_cam'] = self.coord_loss(joint_param_cam, targets['fit_joint_cam'], is_valid_fit[:,:,None])
                    loss['pose_prior'] = self.gmm(pose_param) * 1e-4
                else:
                    loss['pose_param'] = self.param_loss(pose_param, targets['pose_param'], is_valid_fit)
                    loss['shape_param'] = self.param_loss(shape_param, targets['shape_param'], is_valid_fit)
                    loss['joint_orig_cam'] = self.coord_loss(joint_param_cam, targets['orig_joint_cam'], orig_joint_valid * is_3D)
                    loss['joint_fit_cam'] = self.coord_loss(joint_param_cam, targets['fit_joint_cam'], is_valid_fit[:,:,None])

            return loss

        # evaluation
        else:
            # obtain root joint depth and camera-space coordinates
            with torch.no_grad():
                mesh_img = self.bb2img(mesh_lixel_img.clone(), info['bb2img_trans'])
                joint_img = self.bb2img(joint_img_from_mesh.clone(), info['bb2img_trans'])
                
                if cfg.stage == 'lixel':
                    root_joint_depth = info['root_joint_depth']
                    root_img = torch.cat((joint_img[:,self.root_joint_idx,None,:2], root_joint_depth[:,None,None]),2)
                    root_cam = self.pixel2cam(root_img, info['focal'], info['princpt'])[:,0,:]
                else:
                    root_cam = self.estimate_root(joint_img, joint_param_cam, info['focal'], info['princpt'])
                    mesh_param_cam += root_cam[:,None,:]
                    mesh_param_cam_reg = mesh_param_cam.clone()
                    root_joint_depth = root_cam[:,2]

                mesh_img[:,:,2] += root_joint_depth[:,None]
                joint_img[:,:,2] += root_joint_depth[:,None]
                mesh_lixel_cam = self.pixel2cam(mesh_img, info['focal'], info['princpt'])
                joint_lixel_cam = self.pixel2cam(joint_img, info['focal'], info['princpt'])
                
            if cfg.stage == 'param' and cfg.optimize_param:
                joint_trunc = (joint_img_from_mesh[:,:,0] > 2) * (joint_img_from_mesh[:,:,0] < cfg.output_hm_shape[2]-2) * \
                              (joint_img_from_mesh[:,:,1] > 2) * (joint_img_from_mesh[:,:,1] < cfg.output_hm_shape[1]-2) * \
                              (joint_img_from_mesh[:,:,2] > 2) * (joint_img_from_mesh[:,:,2] < cfg.output_hm_shape[0]-2)
                mesh_trunc = (mesh_lixel_img[:,:,0] > 2) * (mesh_lixel_img[:,:,0] < cfg.output_hm_shape[2]-2) * \
                             (mesh_lixel_img[:,:,1] > 2) * (mesh_lixel_img[:,:,1] < cfg.output_hm_shape[1]-2) * \
                             (mesh_lixel_img[:,:,2] > 2) * (mesh_lixel_img[:,:,2] < cfg.output_hm_shape[0]-2)

                targets_from_lixel = {
                    'mesh_img': mesh_lixel_img.detach(),
                    'mesh_vis': mesh_vis.detach(),
                    'mesh_cam': mesh_lixel_cam,
                    'joint_cam': joint_lixel_cam,
                    'mesh_trunc': mesh_trunc,
                    'joint_trunc': joint_trunc,
                }

                # optimize smpl parameters
                root_cam = Variable(root_cam, requires_grad=True)
                pose_global = Variable(pose_param[:,:3].detach(), requires_grad=True)
                pose = Variable(pose_param[:,3:].detach(), requires_grad=True)
                shape = Variable(torch.from_numpy(np.zeros((bs, 10))).float().cuda(), requires_grad=True)

                losses = []
                params = [root_cam, pose_global, pose, shape]
                mesh_param_cam = self.optimize_param(
                    params, 1e-1, 100, targets_from_lixel, losses, [root_cam, pose_global], prior=False)
                mesh_param_cam = self.optimize_param(
                    params, 1e-2, 60, targets_from_lixel, losses, [pose])
                mesh_param_cam = self.optimize_param(
                    params, 1e-3, 50, targets_from_lixel, losses, [root_cam, pose_global, pose])
                mesh_param_cam = self.optimize_param(
                    params, 1e-1, 10, targets_from_lixel, losses, [shape], prior=False)

            # test output
            out = {}
            out['img'] = inputs['img']
            for k in info:
                out[k] = info[k]
            for k in targets:
                out[k] = targets[k]
            
            out['joint_lixel_img'] = joint_lixel_img
            out['joint_lixel_cam'] = joint_lixel_cam
            out['mesh_lixel_img'] = mesh_lixel_img
            out['mesh_lixel_cam'] = mesh_lixel_cam
            if cfg.predict_vis:
                out['joint_vis'] = joint_vis
                out['mesh_vis'] = mesh_vis
            if cfg.stage == 'param':
                out['mesh_param_cam_reg'] = mesh_param_cam_reg
                out['mesh_param_cam'] = mesh_param_cam
                out['root_cam'] = root_cam
            return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


def get_model(vertex_num, joint_num, mode):
    pose_backbone = ResNetBackbone(cfg.resnet_type)
    pose_net = PoseNet(joint_num)
    pose2feat = Pose2Feat(joint_num)
    mesh_backbone = ResNetBackbone(cfg.resnet_type)
    mesh_net = MeshNet(vertex_num)
    param_regressor = ParamRegressor(joint_num, vertex_num)

    if mode == 'train':
        pose_backbone.init_weights()
        pose_net.apply(init_weights)
        pose2feat.apply(init_weights)
        mesh_backbone.init_weights()
        mesh_net.apply(init_weights)
        param_regressor.apply(init_weights)

    model = Model(pose_backbone, pose_net, pose2feat, mesh_backbone, mesh_net, param_regressor)
    return model
