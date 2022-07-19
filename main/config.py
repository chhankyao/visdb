import os
import os.path as osp
import sys
import numpy as np


class Config:  
    ## dataset
    trainset_3d = ['Human36M', 'MuCo', 'PW3D'] # 'Human36M', 'MuCo', 'PW3D'
    trainset_2d = ['MSCOCO'] # 'MSCOCO'
    testset = 'PW3D' # Human36M, MSCOCO, PW3D
    skip_h36m = 1
    skip_muco = 1
    skip_coco = 1

    ## model setting
    backbone = 'resnet'
    resnet_type = 50 # 50, 101, 152
    add_head_top = True
    predict_vis = True
    use_dp = True
    
    ## input, output
    exp_name = 'exp_visdb'
    input_img_shape = (256, 256) 
    output_hm_shape = (64, 64, 64)
    bbox_3d_size = 2
    sigma = 2.5
    no_augment = False
    occ_augment = True

    ## training config
    lr_dec_epoch = [10,12]
    end_epoch = 6
    lr_dec_factor = 10
    train_batch_size = 16

    lr = 1e-4
    w_joint = 1.0
    w_joint_fit = 1.0
    w_mesh_joint = 1.0
    w_mesh_joint_fit = 1.0
    w_mesh_fit = 1.0
    w_mesh_normal = 0.1
    w_mesh_edge = 1.0
    w_dps = 0.1
    w_vis = 0.5

    ## testing config
    test_batch_size = 16
    use_gt_info = False
    optimize_param = True
    optimize_depth = False
    pw3d_occ = False
    
    ## others
    num_thread = 30
    gpu_ids = '0'
    num_gpus = 1
    stage = 'lixel' # lixel, param
    continue_train = False
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join('/tmp', 'data')
    output_dir = osp.join('/tmp', 'results')
    model_dir = osp.join(output_dir, exp_name, 'model_dump')
    log_dir = osp.join(output_dir, exp_name, 'log')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    gmm_path = osp.join(root_dir, 'common', 'gmm_08.pkl')
    dp_path = osp.join(root_dir, 'common')
    
    def set_args(self, gpu_ids, stage='lixel', continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.stage = stage
        # extend training schedule
        if self.stage == 'param':
            self.lr_dec_epoch = [x+5 for x in self.lr_dec_epoch]
            self.end_epoch = self.end_epoch + 5
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))
        
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.root_dir, 'data'))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.root_dir, 'data', cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.root_dir, 'data', cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.root_dir, 'data', cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.log_dir)
