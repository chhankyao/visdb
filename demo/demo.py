import sys
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image
from utils.transforms import pixel2cam, cam2pixel
from utils.vis import vis_mesh, save_obj, render_mesh, front_renderer, side_renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

sys.path.insert(0, cfg.smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--stage', type=str, dest='stage')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--model_path', type=str, dest='model_path', default='/tmp2/chunhan/results/visdb/exp_visdb/model_dump/')
    parser.add_argument('--input_path', type=str, dest='input_path')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, args.stage)
cudnn.benchmark = True


# SMPL model
joint_num = 30 # original: 24. manually add nose, L/R eye, L/R ear, Head_top
vertex_num = 6890
smpl_layer = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
face = smpl_layer.th_faces.numpy()

# snapshot load
model_path = args.model_path + 'snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# read input image
img_raw = cv2.imread(args.input_path)
height, width = img_raw.shape[:2]

# camera parameters
focal = (2000, 2000)
princpt = (width/2, height/2)
root_depth = 3 # in meters

# affine transform
bbox_orig = [0, 0, width, height] # xmin, ymin, width, height      ##### change bbox if known
bbox = process_bbox(bbox_orig, width, height, 'test', 'MSCOCO')
img, _, img2bb_trans, bb2img_trans = generate_patch_image(img_raw, np.zeros(img_raw.shape), 
                                                          bbox, 1.0, 0.0, False, cfg.input_img_shape)

# prepare inputs
transform = transforms.ToTensor()
img = transform(img.astype(np.float32))/255
img = img.cuda()[None,:,:,:]

inputs = {'img': img}
targets = {}
meta_info = {
    'bb2img_trans': torch.from_numpy(bb2img_trans)[None,:,:].cuda(), 
    'focal': torch.tensor(focal)[None,:].cuda(), 
    'princpt': torch.tensor(princpt)[None,:].cuda(), 
    'root_joint_depth': torch.tensor([root_depth]).cuda()
}

out = model(inputs, targets, meta_info, 'test')

# outputs
mesh_lixel_img = out['mesh_lixel_img'][0].detach().cpu().numpy()
mesh_lixel_cam = out['mesh_lixel_cam'][0].detach().cpu().numpy()

if cfg.stage == 'param':
    mesh_param_cam = out['mesh_param_cam'][0].detach().cpu().numpy()
    root_depth = out['root_cam'][0,2].detach().cpu().numpy()
    print('Estimated root depth:', root_depth)
else:
    mesh_param_cam = mesh_lixel_cam.copy()
    
if cfg.predict_vis:
    mesh_vis = out['mesh_vis'][0].detach().cpu().numpy() >= 0.5
    mesh_trunc = mesh_vis[:,0] * mesh_vis[:,1]
    mesh_occ = mesh_vis[:,2] * mesh_trunc
else:
    mesh_trunc = (mesh_lixel_img[:,0] > 2) * (mesh_lixel_img[:,0] < cfg.output_hm_shape[2]-3) * \
                 (mesh_lixel_img[:,1] > 2) * (mesh_lixel_img[:,1] < cfg.output_hm_shape[1]-3) * \
                 (mesh_lixel_img[:,2] > 2) * (mesh_lixel_img[:,2] < cfg.output_hm_shape[0]-3)
    mesh_occ = mesh_trunc
    
output_prefix = args.input_path.replace('.jpg', '')
    
# cropped input image
cv2.imwrite(output_prefix + '_crop.jpg', img[0].cpu().numpy().transpose(1,2,0)*255.)

# render mesh from lixel
rendered_img = render_mesh(img_raw.copy(), mesh_lixel_cam, face, {'focal': focal, 'princpt': princpt})
cv2.imwrite(output_prefix + '_lixel.jpg', rendered_img)

# render mesh from param
rendered_img = render_mesh(img_raw.copy(), mesh_param_cam, face, {'focal': focal, 'princpt': princpt})
cv2.imwrite(output_prefix + '_param.jpg', rendered_img)

# mesh visibility
plt.figure()
plt.scatter(mesh_lixel_cam[:,0][mesh_trunc>0], -mesh_lixel_cam[:,1][mesh_trunc>0], s=1, c='purple')
plt.scatter(mesh_lixel_cam[:,0][mesh_trunc==0], -mesh_lixel_cam[:,1][mesh_trunc==0], s=0.1, c='orange')
plt.axis('off')
plt.savefig(output_prefix + '_mesh_trunc.jpg', bbox_inches='tight')

# mesh visibility
plt.figure()
plt.scatter(mesh_lixel_cam[:,0][mesh_occ>0], -mesh_lixel_cam[:,1][mesh_occ>0], s=1, c='purple')
plt.scatter(mesh_lixel_cam[:,0][mesh_occ==0], -mesh_lixel_cam[:,1][mesh_occ==0], s=0.1, c='orange')
plt.axis('off')
plt.savefig(output_prefix + '_mesh_occ.jpg', bbox_inches='tight')

# mesh visibility
plt.figure()
plt.scatter(mesh_lixel_cam[:,2][mesh_occ>0], -mesh_lixel_cam[:,1][mesh_occ>0], s=1, c='purple')
plt.scatter(mesh_lixel_cam[:,2][mesh_occ==0], -mesh_lixel_cam[:,1][mesh_occ==0], s=0.1, c='orange')
plt.axis('off')
plt.savefig(output_prefix + '_mesh_occ_side.jpg', bbox_inches='tight')

# render mesh with visibility
# mesh_param_cam[:,:2] *= 0.5
mesh_param_cam -= np.mean(mesh_param_cam, 0)
mesh_lixel_cam -= np.mean(mesh_lixel_cam, 0)

verts_color = torch.zeros(vertex_num, 3).cuda()
verts_color[out['mesh_vis'][0,:,2]>=0.5,:] = torch.tensor([0.1, 1.0, 1.0]).cuda()
verts_color[out['mesh_vis'][0,:,2]<0.5,:] = torch.tensor([1.0, 0.1, 0.1]).cuda()
mesh_vis = Meshes(verts=[torch.from_numpy(mesh_lixel_cam).cuda()],   
                  faces=[torch.from_numpy(face).cuda()],
                  textures=Textures(verts_rgb=verts_color[None,:,:]))
mesh_front_view = front_renderer(mesh_vis)[0].cpu().numpy()[::-1,::-1]
mesh_side_view = side_renderer(mesh_vis)[0].cpu().numpy()[::-1,::-1]
cv2.imwrite(output_prefix + '_vis_front.jpg', mesh_front_view*255.)
cv2.imwrite(output_prefix + '_vis_side.jpg', mesh_side_view*255.)
