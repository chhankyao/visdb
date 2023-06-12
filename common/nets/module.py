import torch
import torch.nn as nn
from torch.nn import functional as F
import torchgeometry as tgm
from nets.layer import make_conv_layers, make_deconv_layers, make_conv1d_layers, make_linear_layers
from config import cfg


class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        if cfg.predict_vis:
            self.conv_vx = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_vy = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_vz_1 = make_conv1d_layers([2048, 256], kernel=1, stride=1, padding=0)
            self.conv_vz_2 = make_conv1d_layers([256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat) # img_feat: bs x 256 x 64 x 64
        # x axis
        img_feat_x = img_feat_xy.mean((2)) # bs x 256 x 64
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)        
        # y axis
        img_feat_y = img_feat_xy.mean((3)) # bs x 256 x 64
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)       
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None] # bs x 2048 x 1
        heatmap_z = self.conv_z_1(img_feat_z).view(-1, 256, cfg.output_hm_shape[0]) # bs x 256 x 64
        heatmap_z = self.conv_z_2(heatmap_z)
        coord_z = self.soft_argmax_1d(heatmap_z)
        # vis
        joint_vis = None
        if cfg.predict_vis:
            trunc_x = self.conv_vx(img_feat_x).mean(2, keepdim=True)
            trunc_y = self.conv_vy(img_feat_y).mean(2, keepdim=True)
            occ = self.conv_vz_1(img_feat_z)
            occ = self.conv_vz_2(occ)
            joint_vis = torch.cat((trunc_x, trunc_y, occ), 2)
            joint_vis = torch.sigmoid(joint_vis)
        joint_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return joint_coord, joint_vis

    
class Pose2Feat(nn.Module):
    def __init__(self, joint_num):
        super(Pose2Feat, self).__init__()
        self.joint_num = joint_num
        self.conv = make_conv_layers([64+joint_num*cfg.output_hm_shape[0],64])
        if cfg.predict_vis:
            self.fc_vis = make_linear_layers([joint_num*3,64])
            self.conv_vis = make_conv_layers([128,64])

    def forward(self, img_feat, joint_heatmap, joint_vis=None):
        joint_heatmap = joint_heatmap.view(-1,self.joint_num*cfg.output_hm_shape[0],cfg.output_hm_shape[1],cfg.output_hm_shape[2])
        feat = torch.cat((img_feat, joint_heatmap),1)
        feat = self.conv(feat)
        if cfg.predict_vis:
            feat_vis = self.fc_vis(joint_vis.view(-1, self.joint_num*3))[:,:,None,None]
            feat = torch.cat((feat, feat_vis.repeat(1, 1, cfg.output_hm_shape[1], cfg.output_hm_shape[2])),1)
            feat = self.conv_vis(feat)
        return feat

    
class MeshNet(nn.Module):
    def __init__(self, vertex_num):
        super(MeshNet, self).__init__()
        self.vertex_num = vertex_num
        self.deconv = make_deconv_layers([2048, 256, 256, 256])
        self.conv_x = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_y = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.conv_z_1 = make_conv1d_layers([2048, 256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
        self.conv_z_2 = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        if cfg.predict_vis:
            self.conv_vx = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_vy = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_vz_1 = make_conv1d_layers([2048, 256], kernel=1, stride=1, padding=0)
            self.conv_vz_2 = make_conv1d_layers([256, self.vertex_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        img_feat_xy = self.deconv(img_feat) # bs x 256 x 64 x 64
        # x axis
        img_feat_x = img_feat_xy.mean((2)) # bs x 256 x 64
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)       
        # y axis
        img_feat_y = img_feat_xy.mean((3)) # bs x 256 x 64
        heatmap_y = self.conv_y(img_feat_y)
        coord_y = self.soft_argmax_1d(heatmap_y)       
        # z axis
        img_feat_z = img_feat.mean((2,3))[:,:,None]
        heatmap_z = self.conv_z_1(img_feat_z).view(-1, 256, cfg.output_hm_shape[0]) # bs x 256 x 64
        heatmap_z = self.conv_z_2(heatmap_z)
        coord_z = self.soft_argmax_1d(heatmap_z)
        # vis
        vert_vis = None
        if cfg.predict_vis:
            trunc_x = self.conv_vx(img_feat_x).mean(2, keepdim=True)
            trunc_y = self.conv_vy(img_feat_y).mean(2, keepdim=True)
            occlusion = self.conv_vz_1(img_feat_z)
            occlusion = self.conv_vz_2(occlusion)
            vert_vis = torch.cat((trunc_x, trunc_y, occlusion), 2)
            vert_vis = torch.sigmoid(vert_vis)
        vert_coord = torch.cat((coord_x, coord_y, coord_z), 2)
        return vert_coord, vert_vis

    
class ParamRegressor(nn.Module):
    def __init__(self, joint_num, vertex_num):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        self.vert_num = vertex_num
        self.feat_dim = 512       
        if cfg.predict_vis:
            self.fc_joint = make_linear_layers([self.joint_num*6, 512, 320], use_bn=True)
            self.fc_vert_x = make_linear_layers([self.vert_num*2, 64, 64], use_bn=True)
            self.fc_vert_y = make_linear_layers([self.vert_num*2, 64, 64], use_bn=True)
            self.fc_vert_z = make_linear_layers([self.vert_num*2, 64, 64], use_bn=True)
        else:
            self.fc = make_linear_layers([self.joint_num*3, 1024, 512], use_bn=True)
        self.fc_pose = make_linear_layers([self.feat_dim, 24*6], relu_final=False) # body joint orientation
        self.fc_shape = make_linear_layers([self.feat_dim, 10], relu_final=False) # shape parameter

    def rot6d_to_rotmat(self,x):
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, joint_3d, vert_3d, joint_vis=None, vert_vis=None):
        batch_size = joint_3d.shape[0]          
        if cfg.predict_vis:
            feat_joint = self.fc_joint(torch.cat((joint_3d, joint_vis), 2).view(-1, self.joint_num*6))
            feat_vert_x = self.fc_vert_x(torch.cat((vert_3d[:,:,0], vert_vis[:,:,0]), 1))
            feat_vert_y = self.fc_vert_y(torch.cat((vert_3d[:,:,1], vert_vis[:,:,1]), 1))
            feat_vert_z = self.fc_vert_z(torch.cat((vert_3d[:,:,2], vert_vis[:,:,2]), 1))
            feat = torch.cat((feat_joint, feat_vert_x, feat_vert_y, feat_vert_z), 1)
        else:
            feat = self.fc(joint_3d.view(-1, self.joint_num*3))
        pose = self.fc_pose(feat)
        pose = self.rot6d_to_rotmat(pose)
        pose = torch.cat([pose, torch.zeros((pose.shape[0], 3, 1)).cuda().float()], 2)
        pose = tgm.rotation_matrix_to_angle_axis(pose).reshape(batch_size, -1)        
        shape = self.fc_shape(feat)
        return pose, shape
