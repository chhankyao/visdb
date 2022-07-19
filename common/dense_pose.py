import os
import os.path as osp
import cv2
import numpy as np
from scipy.io import loadmat
import torch
from config import cfg
from operator import itemgetter
from config import cfg


class DensePose():
    def __init__(self, dp_path):
        # 0      = Background
        # 1, 2   = Torso
        # 3      = Right Hand
        # 4      = Left Hand
        # 5      = Right Foot
        # 6      = Left Foot
        # 7, 9   = Upper Leg Right
        # 8, 10  = Upper Leg Left
        # 11, 13 = Lower Leg Right
        # 12, 14 = Lower Leg Left
        # 15, 17 = Upper Arm Left
        # 16, 18 = Upper Arm Right
        # 19, 21 = Lower Arm Left
        # 20, 22 = Lower Arm Right
        # 23, 24 = Head

        part24_to_14 = {1:1, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:6, 10:7, 11:8, 12:9, 13:8, 14:9, 
                        15:10, 16:11, 17:10, 18:11, 19:12, 20:13, 21:12, 22:13, 23:14, 24:14}
        
        ALP_UV = loadmat(osp.join(dp_path, 'UV_Processed.mat'))
        U = ALP_UV['All_U_norm']                 # (7829, 1)
        V = ALP_UV['All_V_norm']                 # (7829, 1)
        verts = ALP_UV["All_vertices"][0] - 1    # (1, 7829), range=(0, 6889)
        faces = ALP_UV['All_Faces'] - 1          # (13774, 3), range=(0, 7828)
        face_indices = ALP_UV['All_FaceIndices'] # (13774, 1), range=(1, 24)
        
        self.Index_Symmetry_List = [1,2,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23]
        self.UV_symmetry_transformations = loadmat(osp.join(dp_path, 'UV_symmetry_transforms.mat'))
        
        vert_parts = np.zeros(7829)
        for i in range(len(faces)):
            v1, v2, v3 = faces[i, :]
            part = int(face_indices[i])
            vert_parts[v1] = part
            vert_parts[v2] = part
            vert_parts[v3] = part
            
        self.parts = np.zeros(6890)
        self.parts14 = np.zeros(6890)
        for i in range(7829):
            self.parts[verts[i]] = vert_parts[i]
            self.parts14[verts[i]] = part24_to_14[vert_parts[i]]
            
        self.res = 30
        self.UV_map = np.zeros((24, self.res, self.res), dtype=np.int32)
        for p in range(24):
            indices = np.where(vert_parts == p+1)[0]
            for u in range(self.res):
                u_verts = U[indices, 0]
                for v in range(self.res):
                    v_verts = V[indices, 0]
                    err = (u_verts - u/(self.res-1))**2 + (v_verts - v/(self.res-1))**2
                    self.UV_map[p, u, v] = int(verts[indices[np.argmin(err)]])
    
    def flip_iuv(self, iuv):
        iuv_flip = np.zeros(iuv.shape)
        for i in range(24):
            mask = np.where(iuv[:,:,0] == (i+1))
            if len(mask[0]) > 0:
                iuv_flip[:,:,0][mask] = self.Index_Symmetry_List[i]
                U_loc = iuv[:,:,1][mask]
                V_loc = iuv[:,:,2][mask]
                iuv_flip[:,:,1][mask] = self.UV_symmetry_transformations['U_transforms'][0,i][V_loc, U_loc]*255.
                iuv_flip[:,:,2][mask] = self.UV_symmetry_transformations['V_transforms'][0,i][V_loc, U_loc]*255.
        return iuv_flip
        
    def dense_point_corr(self, iuv):
        part_resized = cv2.resize(iuv[:,:,0], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        uv_resized = cv2.resize(iuv[:,:,1:], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        indices_y, indices_x = np.where(part_resized > 0)
        part = (part_resized[indices_y, indices_x] - 1).astype(np.int32)
        uv = (uv_resized[indices_y, indices_x, :]/255. * (self.res-1)).clip(0,self.res-1).astype(np.int32)
        vertices = self.UV_map[part, uv[:,0], uv[:,1]]
        unique_vertices = np.unique(vertices)
        for v in unique_vertices:
            idx = np.where(vertices == v)
            indices_x[idx] = np.mean(indices_x[idx])
            indices_y[idx] = np.mean(indices_y[idx])
        dps = np.zeros([6890,3], dtype=np.float32)
        dps[vertices,0] = indices_x*1. / np.shape(uv_resized)[1] * cfg.output_hm_shape[2]
        dps[vertices,1] = indices_y*1. / np.shape(uv_resized)[0] * cfg.output_hm_shape[1]
        dps[vertices,2] = 1
        return dps
    
    
dp = DensePose(cfg.dp_path)
